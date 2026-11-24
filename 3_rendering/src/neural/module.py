from pathlib import Path

import torch
from pytorch_lightning import Callback, LightningModule
from torchvision.utils import make_grid, save_image

from .model import Discriminator, UNetGenerator, weights_init


class PhongGANModule(LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        model_cfg = cfg["model"]
        train_cfg = cfg["train"]
        image_size = cfg["data"]["image_size"]
        embed_cfg = model_cfg["param_embed"]

        self.z_dim = model_cfg["z_dim"]
        self.z_mode = model_cfg.get("z_mode", "random")
        self.critic_iter = train_cfg["critic_iter"]
        self.gp_lambda = train_cfg["gp_lambda"]
        self.adv_lambda = train_cfg.get("adv_lambda", 1.0)
        self.l1_lambda = train_cfg.get("l1_lambda", 0.0)
        self.grad_lambda = train_cfg.get("grad_lambda", 0.0)
        self.recon_masked = train_cfg.get("recon_masked", False)
        self.bg_weight = train_cfg.get("bg_weight", 1.0)
        self.base_lr = train_cfg["lr"]
        self.critic_burn_in_steps = train_cfg.get("critic_burn_in_steps", 0)
        self.lr_warmup_steps = train_cfg.get("lr_warmup_steps", 0)

        gen_cfg = model_cfg["generator"]
        self.generator = UNetGenerator(
            param_dim=model_cfg["param_dim"],
            z_dim=model_cfg["z_dim"],
            image_size=image_size,
            base=gen_cfg.get("base", 40),
            groups=gen_cfg.get("groups", 8),
        )
        self.critic = Discriminator(
            param_dim=model_cfg["param_dim"],
            image_size=image_size,
            channels=model_cfg["discriminator"]["channels"],
            fc_dim=model_cfg["discriminator"]["fc_dim"],
            embed_hidden=embed_cfg["hidden"],
            embed_out=embed_cfg["out"],
        )
        self.critic.apply(weights_init)

        self.automatic_optimization = False
        self._fixed_z = None
        self.register_buffer("_train_steps", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------
    def sample_z(self, n: int, device) -> torch.Tensor:
        """Szum wejściowy generatora."""
        if self.z_dim == 0:
            return torch.zeros(n, 0, device=device)
        if self.z_mode == "zero":
            return torch.zeros(n, self.z_dim, device=device)
        return torch.randn(n, self.z_dim, device=device)

    def get_fixed_z(self, n: int, device) -> torch.Tensor:
        """Stały szum dla siatek próbek (te same wejścia między epokami)."""
        if self._fixed_z is None or self._fixed_z.shape[0] != n:
            if self.z_mode == "zero":
                self._fixed_z = torch.zeros(n, self.z_dim)
            else:
                self._fixed_z = torch.randn(n, self.z_dim)
        return self._fixed_z.to(device)

    def gradient_penalty(self, real, fake, params):
        batch = real.size(0)
        alpha = torch.rand(batch, 1, 1, 1, device=real.device)
        interp = (real * alpha + fake * (1 - alpha)).requires_grad_(True)
        pred = self.critic(interp, params)
        grads = torch.autograd.grad(
            outputs=pred,
            inputs=interp,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

    @staticmethod
    def foreground_mask(real: torch.Tensor) -> torch.Tensor:
        """Maska obiektu z obrazu referencyjnego (real w [-1, 1])."""
        return (real.max(dim=1, keepdim=True).values > -0.98).float()

    @staticmethod
    def gradient_loss(
        fake: torch.Tensor, real: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Porównuje pochodne poziome i pionowe obrazu — karze za rozmyte krawędzie."""
        dx_fake = fake[:, :, :, 1:] - fake[:, :, :, :-1]
        dx_real = real[:, :, :, 1:] - real[:, :, :, :-1]
        dy_fake = fake[:, :, 1:, :] - fake[:, :, :-1, :]
        dy_real = real[:, :, 1:, :] - real[:, :, :-1, :]
        if mask is None:
            return (dx_fake - dx_real).abs().mean() + (dy_fake - dy_real).abs().mean()
        mdx = mask[:, :, :, 1:]
        mdy = mask[:, :, 1:, :]
        gx = ((dx_fake - dx_real).abs() * mdx).sum() / mdx.sum().clamp_min(1)
        gy = ((dy_fake - dy_real).abs() * mdy).sum() / mdy.sum().clamp_min(1)
        return gx + gy

    def reconstruction_terms(self, fake: torch.Tensor, real: torch.Tensor):
        """(L1, gradient) z opcjonalną normalizacją po obiekcie i tle."""
        if not self.recon_masked:
            mask = self.foreground_mask(real)
            l1 = ((fake - real).abs() * (1.0 + 9.0 * mask)).mean()
            return l1, self.gradient_loss(fake, real)
        mask = self.foreground_mask(real)
        diff = (fake - real).abs()
        fg = (diff * mask).sum() / mask.sum().clamp_min(1)
        bg = (diff * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1)
        l1 = fg + self.bg_weight * bg
        return l1, self.gradient_loss(fake, real, mask=mask)

    def _current_lr_scale(self) -> float:
        """Liniowe rozgrzanie lr: 0 -> 1 w ciągu lr_warmup_steps kroków."""
        if self.lr_warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self._train_steps) / self.lr_warmup_steps)

    def _apply_lr(self, optimizer):
        scale = self._current_lr_scale()
        for group in optimizer.param_groups:
            group["lr"] = self.base_lr * scale

    def training_step(self, batch, batch_idx):
        real, params, _ = batch
        opt_c, opt_g = self.optimizers()

        self._train_steps += 1
        self._apply_lr(opt_c)
        self._apply_lr(opt_g)

        for _ in range(self.critic_iter):
            z = self.sample_z(real.size(0), real.device)
            with torch.no_grad():
                fake = self.generator(z, params)
            gp = self.gradient_penalty(real, fake, params)
            c_loss = (
                self.critic(fake, params).mean()
                - self.critic(real, params).mean()
                + self.gp_lambda * gp
            )
            opt_c.zero_grad(set_to_none=True)
            self.manual_backward(c_loss)
            opt_c.step()

        burning_in = int(self._train_steps) <= self.critic_burn_in_steps
        if not burning_in:
            z = self.sample_z(real.size(0), real.device)
            fake = self.generator(z, params)
            adv_loss = -self.critic(fake, params).mean()
            l1_loss, grad_loss = self.reconstruction_terms(fake, real)
            g_loss = (
                self.adv_lambda * adv_loss
                + self.l1_lambda * l1_loss
                + self.grad_lambda * grad_loss
            )

            opt_g.zero_grad(set_to_none=True)
            self.manual_backward(g_loss)
            opt_g.step()

        self.log("train/critic_loss", c_loss, prog_bar=True)
        self.log("train/gradient_penalty", gp)
        self.log("train/lr_scale", self._current_lr_scale())
        if not burning_in:
            self.log("train/gen_loss", g_loss, prog_bar=True)
            self.log("train/gen_adv_loss", adv_loss)
            self.log("train/gen_adv_weighted", self.adv_lambda * adv_loss)
            self.log("train/gen_l1_loss", l1_loss, prog_bar=True)
            self.log("train/gen_grad_loss", grad_loss)
        return c_loss

    def validation_step(self, batch, batch_idx):
        real, params, _ = batch
        with torch.no_grad():
            z = self.sample_z(real.size(0), real.device)
            fake = self.generator(z, params)
        l1, _ = self.reconstruction_terms(fake, real)
        self.log("val/l1", l1, sync_dist=True)

    def configure_optimizers(self):
        t = self.cfg["train"]
        betas = (t["beta1"], t["beta2"])
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=t["lr"], betas=betas)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=t["lr"], betas=betas)
        return [opt_c, opt_g]
