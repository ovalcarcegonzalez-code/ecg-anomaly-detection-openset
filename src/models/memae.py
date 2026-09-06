"""
Arquitectura MemAE híbrida para detección de anomalías en ECG.

El modelo combina dos fuentes de información:
  - Morfología del latido (rama convolucional 1D).
  - Contexto de ritmo mediante intervalos RR (rama densa).

Ambas se fusionan en un espacio latente que atraviesa un módulo de memoria
(Gong et al., 2019). Este módulo impide que el decoder reconstruya libremente:
solo puede combinar prototipos de normalidad aprendidos durante el
entrenamiento, lo que amplifica el error de reconstrucción sobre entradas
anómalas.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    LATENT_DIM,
    LONGITUD_SENAL,
    MEM_DIM,
    N_RR_FEATURES,
    SHRINK_THRES,
)


class MemoryModule(nn.Module):

    def __init__(
        self,
        mem_dim: int = MEM_DIM,
        latent_dim: int = LATENT_DIM,
        shrink_thres: float = SHRINK_THRES,
    ) -> None:
        super().__init__()
        self.mem_dim = mem_dim
        self.latent_dim = latent_dim
        self.shrink_thres = shrink_thres
        self.memory = nn.Parameter(torch.randn(mem_dim, latent_dim) * 0.05)

    @staticmethod
    def _hard_shrink_relu(
        x: torch.Tensor, lambd: float = 0.0, epsilon: float = 1e-12
    ) -> torch.Tensor:
        #Anula los pesos por debajo de lamda, manteniendo el resto
        return (F.relu(x - lambd) * x) / (torch.abs(x - lambd) + epsilon)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Similitud coseno entre la consulta y cada prototipo
        att = F.linear(F.normalize(z, dim=1), F.normalize(self.memory, dim=1))
        att = F.softmax(att, dim=1)

        if self.shrink_thres > 0:
            att = self._hard_shrink_relu(att, lambd=self.shrink_thres)
            att = F.normalize(att, p=1, dim=1)

        z_hat = torch.mm(att, self.memory)
        return z_hat, att


def entropy_loss(att: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    
    return torch.mean(torch.sum(-att * torch.log(att + eps), dim=1))


class MemAEHibrido(nn.Module):


    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        n_rr_features: int = N_RR_FEATURES,
        longitud_senal: int = LONGITUD_SENAL,
        mem_dim: int = MEM_DIM,
        shrink_thres: float = SHRINK_THRES,
    ) -> None:
        super().__init__()

        if longitud_senal % 8 != 0:
            raise ValueError(
                f"longitud_senal debe ser divisible por 8 (tres capas con stride 2); "
                f"recibido: {longitud_senal}"
            )

        self.longitud_senal = longitud_senal
        self.conv_len = longitud_senal // 8
        self.flatten_dim = 64 * self.conv_len

        # Rama de morfología 
        self.encoder_signal = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )

        # Rama de ritmo 
        self.encoder_rr = nn.Sequential(
            nn.Linear(n_rr_features, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )

        # Fusión y memoria
        self.fc_latent = nn.Linear(self.flatten_dim + 8, latent_dim)
        self.memory = MemoryModule(mem_dim, latent_dim, shrink_thres)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim + 8)

        # Decoders 
        self.decoder_signal = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )
        self.decoder_rr = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, n_rr_features),
        )

    def forward(
        self, x_signal: torch.Tensor, x_rr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        b = x_signal.size(0)

        z_sig = self.encoder_signal(x_signal)
        z_rr = self.encoder_rr(x_rr)
        z = self.fc_latent(torch.cat([z_sig.view(b, -1), z_rr], dim=1))

        z_hat, att = self.memory(z)

        dec = self.fc_dec(z_hat)
        dec_sig = dec[:, : self.flatten_dim].view(b, 64, self.conv_len)
        dec_rr = dec[:, self.flatten_dim :]

        out_signal = self.decoder_signal(dec_sig)
        out_rr = self.decoder_rr(dec_rr)
        return out_signal, out_rr, att