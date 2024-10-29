# Standard library imports
import os

# Third-party imports
import warnings
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import traceback

# Custom imports
from ..utils.helpers import setup_logger
from ..utils.helpers import collate_fn

# Constants
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set environment variables
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')

class TripletTrainer:
    def __init__(self, model, train_dataset, val_dataset, rank, world_size):
        self.logger = setup_logger()
        self.logger.info(f"[Rank {rank}] Initializing TripletTrainer")

        self.rank = rank
        self.world_size = world_size
        self.model = model

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=self.rank, shuffle=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            num_workers=2,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=collate_fn
        )

        self.model = self.model.to(DEVICE)
        self.logger.info(f"[Rank {rank}] Model moved to device {self.rank}")

        model_path = 'triplet_loss/data/best_triplet_model.pth'
        if os.path.exists(model_path):
            self.logger.info(f"[Rank {rank}] Loading existing model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        try:
            self.model = DistributedDataParallel(self.model, device_ids=None, find_unused_parameters=True)
            self.logger.info(f"[Rank {rank}] Model wrapped with DistributedDataParallel")
        except Exception as e:
            self.logger.error(f"[Rank {rank}] Exception during model wrapping: {e}")
            traceback.print_exc()
            raise e

        self.criterion = torch.nn.TripletMarginLoss(margin=1.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.logger.info(f"[Rank {rank}] Optimizer initialized")

        self.val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            num_workers=2,
            shuffle=False,
            sampler=self.val_sampler,
            collate_fn=collate_fn
        )
        
        self.logger.info(f"[Rank {self.rank}] Train loader initialized with {len(self.train_loader)} batches")

        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)

    def train_epoch(self, epoch):
        try:
            self.logger.info(f"[Rank {self.rank}] Starting training epoch {epoch}")
            self.train_loader.sampler.set_epoch(epoch)

            self.model.train()
            total_loss = 0.0
            total_samples = 0
            accumulated_loss = 0.0
            accumulation_steps = 16
            self.optimizer.zero_grad()

            for i, (anchor, positive, negative, protein_type_id, batch_size) in enumerate(tqdm(self.train_loader, desc="Training", disable=(self.rank != 0))):
                anchor = anchor.to(DEVICE)
                positive = positive.to(DEVICE)
                negative = negative.to(DEVICE)
                protein_type_id = protein_type_id.to(DEVICE)

                anchor_out = self.model(anchor, protein_type_id)
                positive_out = self.model(positive, protein_type_id)
                negative_out = self.model(negative, protein_type_id)

                loss = self.criterion(anchor_out, positive_out, negative_out)
                loss = loss / accumulation_steps
                loss.backward()
                accumulated_loss += loss.item() * batch_size

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += accumulated_loss
                    accumulated_loss = 0.0

                total_samples += batch_size

            self.logger.info(f"[Rank {self.rank}] Finished training epoch {epoch}")

            all_losses = [torch.zeros(1).to(DEVICE) for _ in range(self.world_size)]
            all_samples = [torch.zeros(1, dtype=torch.long).to(DEVICE) for _ in range(self.world_size)]
            
            dist.all_gather(all_losses, torch.tensor([total_loss]).to(DEVICE))
            dist.all_gather(all_samples, torch.tensor([total_samples]).to(DEVICE))

            total_loss = sum(loss.item() for loss in all_losses)
            total_samples = sum(samples.item() for samples in all_samples)

            average_loss = total_loss / total_samples
            return average_loss
        except Exception as e:
            self.logger.error(f"[Rank {self.rank}] Error in train_epoch: {e}")
            traceback.print_exc()
            raise e

    def validate(self):
        try:
            self.logger.info(f"[Rank {self.rank}] Starting validation")
            self.model.eval()
            total_loss = 0
            total_samples = 0
            with torch.no_grad():
                for anchor, positive, negative, protein_type_id, batch_size in tqdm(self.val_loader, desc="Validating", disable=(self.rank != 0)):
                    anchor = anchor.to(DEVICE)
                    positive = positive.to(DEVICE)
                    negative = negative.to(DEVICE)
                    protein_type_id = protein_type_id.to(DEVICE)

                    anchor_out = self.model(anchor, protein_type_id)
                    positive_out = self.model(positive, protein_type_id)
                    negative_out = self.model(negative, protein_type_id)

                    loss = self.criterion(anchor_out, positive_out, negative_out)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

            self.logger.info(f"[Rank {self.rank}] Finished validation")

            all_losses = [torch.zeros(1).to(DEVICE) for _ in range(self.world_size)]
            all_samples = [torch.zeros(1, dtype=torch.long).to(DEVICE) for _ in range(self.world_size)]
            
            dist.all_gather(all_losses, torch.tensor([total_loss]).to(DEVICE))
            dist.all_gather(all_samples, torch.tensor([total_samples]).to(DEVICE))

            total_loss = sum(loss.item() for loss in all_losses)
            total_samples = sum(samples.item() for samples in all_samples)

            average_loss = total_loss / total_samples
            return average_loss
        except Exception as e:
            self.logger.error(f"[Rank {self.rank}] Error in validate: {e}")
            traceback.print_exc()
            raise e
