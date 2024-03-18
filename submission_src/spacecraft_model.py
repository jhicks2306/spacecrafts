import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

class SpacecraftModel(L.LightningModule):
    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.example_input_array = [torch.Tensor(1, 3, 512, 640)]
    
    def training_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def forward(self, images):
        self.model.eval()
        preds = self.model(images)
        return preds
    
    def validation_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("valid_loss", loss, prog_bar=True, on_epoch=True)
    
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=(self.learning_rate))
        return optimizer
    