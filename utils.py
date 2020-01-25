import torch
import os

# dataset filenames
dt_files = {'ehr-file': 'cohort-ehrseq.csv',
            'ehr-file-test': 'cohort_test-ehrseq.csv',
            'vocab': 'cohort-vocab.csv'}

# model parameters
model_param = {'num_epochs': 5,
               'batch_size': 128,
               'embedding_size': 100,
               'kernel_size': 5,
               'learning_rate': 0.0001,
               'weight_decay': 1e-5
               }

# length of padded sub-sequences
len_padded = 32

# save the best model
def save_best_model(epoch, model, optimizer, loss, outdir):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, os.path.join(outdir, 'best_model.pt'))
