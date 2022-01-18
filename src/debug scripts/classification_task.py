import torch
import torchdrug as td
from torchdrug import data


from torchdrug import datasets

dataset = datasets.ClinTox("~/molecule-datasets/")
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

from torchdrug import core, models, tasks

model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[256, 256, 256, 256],
                   short_cut=True, batch_norm=True, concat_hidden=True)

task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                criterion="bce", metric=("auprc", "auroc"))

optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     batch_size=1024, gpus=[0])
solver.train(num_epoch=100)

solver.evaluate("valid")

batch = data.graph_collate(valid_set[:8])
pred = task.predict(batch)


#save model
import json

with open("clintox_gin.json", "w") as fout:
    json.dump(solver.config_dict(), fout)
solver.save("clintox_gin.pth")

#load model
with open("clintox_gin.json", "r") as fin:
    solver = core.Configurable.load_config_dict(json.load(fin))
solver.load("clintox_gin.pth")