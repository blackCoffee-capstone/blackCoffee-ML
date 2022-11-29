from modelTesting import HybridRecSystem
from utils import HybridRecDataset
from torch.utils.data import DataLoader, Dataset

def train(
    epoch,
    model,
    device,
    loader,
    optimizer
):
    model.train()

    for _,data in enumerate(loader, 0):
        ratings = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous() ##lastmask
        y_pos_ids = data['target_pos_ids'][:, :-1].to(device, dtype = torch.long)
        y_wholeWord_ids = data['target_wholeWord_ids'][:, :-1].to(device, dtype = torch.long)
        lm_labels = y[:, 1:].clone().detach() ##leftShift
        lm_labels[y[:, 1:] == 0] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        pos_ids = data['source_pos_ids'].to(device, dtype = torch.long)
        wholeWord_ids = data['source_wholeWord_ids'].to(device, dtype = torch.long)
        
        mask = data['source_mask'].to(device, dtype = torch.long)
        
        #print(y_ids,lm_labels)
        
        #outputs = model(input_ids = ids, pos_ids = pos_ids, wholeWord_ids = wholeWord_ids, attention_mask = mask, decoder_input_ids=y_ids, decoder_pos_ids=y_pos_ids, decoder_wholeWord_ids=y_wholeWord_ids, labels=lm_labels)
        outputs = model(input_ids = ids, pos_ids = pos_ids, wholeWord_ids = wholeWord_ids, attention_mask = mask, labels=lm_labels)
        loss = outputs[0]
        
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})
        
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(train_set_path):
