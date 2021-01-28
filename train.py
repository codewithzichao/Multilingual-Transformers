import torch
from preprocessing import *
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
from sklearn.metrics import classification_report
from apex import amp


class Trainer(object):
    def __init__(self, model, fgm, train_loader, dev_loader, test_loader, optimizer, scheduler, loss_fn, save_path, \
                 epochs, writer, max_norm, eval_step_interval, device,load_save=False):
        super(Trainer, self).__init__()

        self.model = model
        self.fgm = fgm
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.epochs = epochs
        self.writer = writer
        self.max_norm = max_norm
        self.eval_step_interval = eval_step_interval
        self.device = device
        self.best_f1 = 0.0
        self.load_save=load_save

        self.model.to(self.device)

        # AMP
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def train(self):
        self.model.train()
        global_step = 1
        self.optimizer.zero_grad()
        for epoch in range(1, self.epochs + 1):

            for idx, batch_data in enumerate(self.train_loader, start=1):
                input_ids, attention_mask, token_type_ids, label, lang_id = batch_data[0], batch_data[1], batch_data[2], \
                                                                            batch_data[3], batch_data[-1]

                logits, lang_logits = self.model(input_ids.to(self.device), attention_mask.to(self.device), \
                                                 token_type_ids.to(self.device))

                loss = self.loss_fn(logits, label.to(self.device))
                #loss/=8
                self.writer.add_scalar("train/loss", loss.item(), global_step=global_step)

                # --------amp----------------
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                # --------amp----------------
                # loss.backward()

                # 对抗训练
                fgm = self.fgm(self.model)
                fgm.attack()
                logits_adv, lang_logits_adv = self.model(input_ids.to(self.device), attention_mask.to(self.device), \
                                                         token_type_ids.to(self.device))
                loss_adv = self.loss_fn(logits_adv, label.to(self.device))
                loss_adv = loss_adv / 8

                # ---------------amp-----------------
                with amp.scale_loss(loss_adv, self.optimizer) as scaled_loss_adv:
                    scaled_loss_adv.backward()
                # ---------------amp------------------
                fgm.restore()
                # 对抗训练

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)

                if idx % 8 == 0:
                    print(datetime.now(), "---",
                          "epoch:{epoch},step:{step},train_loss:{loss}.".format(epoch=epoch, step=idx,
                                                                                loss=loss.item()))
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if idx % self.eval_step_interval == 0:
                        p, r, f1 = self.eval()
                        self.model.train()

                        self.writer.add_scalar("dev/p", p, global_step)
                        self.writer.add_scalar("dev/r", r, global_step)
                        self.writer.add_scalar("dev/f1", f1, global_step)

                        if self.best_f1 < f1:
                            self.best_f1 = f1
                            ckpt_dict = {
                                "model": self.model.state_dict()
                            }
                            torch.save(ckpt_dict, f=self.save_path)

                        print(datetime.now(), "---", \
                              "epoch:{epoch},step:{idx},precision:{p},recall:{r},F1-score:{f1},best_F1:{best_f1}".format(
                                  epoch=epoch, idx=idx, \
                                  p=p, r=r, f1=f1, best_f1=self.best_f1))
                        print("------end evaluating model in dev data------")

                global_step += 1

        self.writer.flush()
        self.writer.close()

    def eval(self):
        self.model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for idx, batch_data in enumerate(self.dev_loader):
                input_ids, attention_mask, token_type_ids, label = batch_data[0], batch_data[1], batch_data[2], \
                                                                   batch_data[3]
                logits, lang_logits = self.model(input_ids.to(self.device), attention_mask.to(self.device),
                                                 token_type_ids.to(self.device))

                y_true.extend(label.cpu().numpy())
                logits = logits.cpu().numpy()
                for item in logits:
                    y_pred.append(np.argmax(item))

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        p = precision_score(y_true, y_pred, average="weighted")
        r = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        cls_report = classification_report(y_true, y_pred)
        print("------start evaluating model in dev data------")
        print(datetime.now(), "---")
        print(cls_report)

        return p, r, f1
