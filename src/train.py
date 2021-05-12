import time
from tqdm.notebook import tqdm

def train_generator(device, G, train_dl, val_dl, opt, criterion, epochs):
  for e in range(epochs):
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    G.train()
    for data in tqdm(train_dl):
      L, ab = data["L"].to(device), data["ab"].to(device)

      preds = G(L)

      loss = criterion(preds, ab)

      opt.zero_grad()
      loss.backward()
      opt.step()

      train_loss_meter.update(loss.item(), L.size(0))

    G.eval()
    for data in tqdm(val_dl):
      L, ab = data["L"].to(device), data["ab"].to(device)

      preds = G(L)

      loss = criterion(preds, ab)

      val_loss_meter.update(loss.item(), L.size(0))

    print(f"Epoch {e + 1}/{epochs}")
    print(f"L1 --- Trn_loss: {train_loss_meter.avg:.4f} --- Val_loss: {val_loss_meter.avg:.4f}")
    torch.save(G.state_dict(), f"./gen_models/{e}_{time.time()}_res18-unet.pt")

def train_model(model, train_dl, val_dl, epochs, display_every=200):
  fixed_val_data = next(iter(val_dl))

  for e in range(epochs):
    loss_meters = init_loss_meters()
    i = 0
    for data in tqdm(train_dl):
      model.setup_input(data)

      model.optimize()

      update_losses(model, loss_meters, data["L"].size(0))

      i += 1
      if i % display_every == 0:
        print(f"\nEpoch {e+1}/{epochs}")
        print(f"Iteration {i}/{len(train_dl)}")
        log_results(loss_meters)

        visualize(model, fixed_val_data, e)

    torch.save(model.D.state_dict(), f"./models/{e}_D_{time.time()}.pt")
    torch.save(model.G.state_dict(), f"./models/{e}_G_{time.time()}.pt")