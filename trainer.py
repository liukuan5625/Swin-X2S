# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import csv
import numpy as np
import torch
import torch.distributed
import torch.nn.parallel
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from utils.misc import AverageMeter, distributed_all_gather


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    run_loss = AverageMeter()
    run_self_loss = AverageMeter()
    run_mutual_loss = AverageMeter()

    self_crit = loss_func["self_crit"]
    mutual_crit = loss_func["mutual_crit"]

    for idx, batch_data in enumerate(loader):
        for param in model.parameters():
            param.grad = None
        data, target = batch_data["img"], batch_data["seg"]

        if mutual_crit:
            data_list = [data[0].cuda(args.rank), data[1].cuda(args.rank)]
            target = target.cuda(args.rank)

            loss = 0
            self_loss_list, mutual_loss_list = [], []
            with (autocast(enabled=args.amp)):
                output1, output2 = model(data_list[0], data_list[1])
                out_list = [output1, output2]

                for i in range(len(out_list)):
                    self_loss = self_crit(out_list[i], target)
                    mutual_loss = 0
                    for j in range(len(out_list)):  # KL divergence
                        if i != j:
                            mutual_end = mutual_crit(F.log_softmax(out_list[i], dim=1), F.softmax(out_list[j], dim=1))
                            mutual_loss += mutual_end
                    loss += (self_loss + mutual_loss) / len(out_list)

                    self_loss_list.append(self_loss.item())
                    mutual_loss_list.append(mutual_loss.item())
                self_loss = torch.mean(torch.tensor(self_loss_list)).cuda(args.rank)
                mutual_loss = torch.mean(torch.tensor(mutual_loss_list)).cuda(args.rank)

        else:
            data = torch.cat([data[0], data[1]], dim=1).cuda(args.rank)
            target = target.cuda(args.rank)

            loss = 0
            with (autocast(enabled=args.amp)):
                output = model(data)
                self_loss = self_crit(output, target)
                mutual_loss = torch.zeros_like(self_loss)
                loss += self_loss

        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if args.distributed:
            is_valid = True
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=is_valid)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
            self_loss_list = distributed_all_gather([self_loss], out_numpy=True, is_valid=is_valid)
            run_self_loss.update(np.mean(np.mean(np.stack(self_loss_list, axis=0), axis=0), axis=0),
                                 n=args.batch_size * args.world_size)
            mutual_loss_list = distributed_all_gather([mutual_loss], out_numpy=True, is_valid=is_valid)
            run_mutual_loss.update(np.mean(np.mean(np.stack(mutual_loss_list, axis=0), axis=0), axis=0),
                                   n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            run_self_loss.update(self_loss.item(), n=args.batch_size)
            run_mutual_loss.update(mutual_loss.item(), n=args.batch_size)

    loss_dic = {"loss": run_loss.avg, "self_loss": run_self_loss.avg, "mutual_loss": run_mutual_loss.avg}
    return loss_dic


def val_epoch(model, loader, epoch, acc_func, loss_func, args, post_label=None, post_pred=None):
    model.eval()

    run_loss = AverageMeter()
    run_self_loss = AverageMeter()
    run_mutual_loss = AverageMeter()
    run_dice = AverageMeter()

    self_crit = loss_func["self_crit"]
    mutual_crit = loss_func["mutual_crit"]

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["img"], batch_data["seg"]

            if mutual_crit:
                data_list = [data[0].cuda(args.rank), data[1].cuda(args.rank)]
                target = target.cuda(args.rank)

                torch.cuda.empty_cache()
                loss = 0
                self_loss_list, mutual_loss_list = [], []
                with autocast(enabled=args.amp):
                    output1, output2 = model(data_list[0], data_list[1])
                    out_list = [output1, output2]

                    for i in range(len(out_list)):
                        self_loss = self_crit(out_list[i], target)

                        mutual_loss = 0
                        for j in range(len(out_list)):  # KL divergence
                            if i != j:
                                mutual_end = mutual_crit(F.log_softmax(out_list[i], dim=1), F.softmax(out_list[j], dim=1))
                                mutual_loss += mutual_end
                        loss += (self_loss + mutual_loss) / len(out_list)

                        self_loss_list.append(self_loss.item())
                        mutual_loss_list.append(mutual_loss.item())
                    self_loss = torch.mean(torch.tensor(self_loss_list)).cuda(args.rank)
                    mutual_loss = torch.mean(torch.tensor(mutual_loss_list)).cuda(args.rank)
                    output = (output1 + output2) / 2

            else:
                data = torch.cat([data[0], data[1]], dim=1).cuda(args.rank)
                target = target.cuda(args.rank)

                torch.cuda.empty_cache()
                loss = 0
                with autocast(enabled=args.amp):
                    output = model(data)
                    self_loss = self_crit(output, target)
                    mutual_loss = torch.zeros_like(self_loss)
                    loss += self_loss
                ###

            output, target = output.cpu(), target.cpu()
            val_labels_list = [i for i in target]
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = [i for i in output]
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            dice_func = acc_func["dice_metric"]
            dice_func.reset()
            dice_func(y_pred=val_output_convert, y=val_labels_convert)
            dice_, dice_not_nans = dice_func.aggregate()
            dice_, dice_not_nans = dice_.cuda(args.rank), dice_not_nans.cuda(args.rank)

            if args.distributed:
                is_valid = True
                loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=is_valid)
                run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                                n=args.batch_size * args.world_size)
                self_loss_list = distributed_all_gather([self_loss], out_numpy=True, is_valid=is_valid)
                run_self_loss.update(np.mean(np.mean(np.stack(self_loss_list, axis=0), axis=0), axis=0),
                                     n=args.batch_size * args.world_size)
                mutual_loss_list = distributed_all_gather([mutual_loss], out_numpy=True, is_valid=is_valid)
                run_mutual_loss.update(np.mean(np.mean(np.stack(mutual_loss_list, axis=0), axis=0), axis=0),
                                       n=args.batch_size * args.world_size)
                ###
                dice_list, not_nans_list = distributed_all_gather(
                    [dice_, dice_not_nans], out_numpy=True, is_valid=is_valid
                )
                for al, nl in zip(dice_list, not_nans_list):
                    run_dice.update(al, n=nl)

            else:
                run_loss.update(loss.item(), n=args.batch_size)
                run_self_loss.update(self_loss.item(), n=args.batch_size)
                run_mutual_loss.update(mutual_loss.item(), n=args.batch_size)
                ###
                run_dice.update(dice_.cpu().numpy(), n=dice_not_nans.cpu().numpy())

    loss_dic = {"loss": run_loss.avg, "self_loss": run_self_loss.avg, "mutual_loss": run_mutual_loss.avg}
    acc_dic = {"dice": np.mean(run_dice.avg)}

    return loss_dic, acc_dic


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        acc_func,
        args,
        scheduler=None,
        start_epoch=0,
        post_label=None,
        post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        loss_dic = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler=scaler,
            epoch=epoch,
            loss_func=loss_func,
            args=args,
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(loss_dic["loss"]),
                "self loss: {:.4f}".format(loss_dic["self_loss"]),
                "mutual loss: {:.4f}".format(loss_dic["mutual_loss"]),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", loss_dic["loss"], epoch)
            writer.add_scalar("self_loss", loss_dic["self_loss"], epoch)
            writer.add_scalar("mutual_loss", loss_dic["mutual_loss"], epoch)

        if epoch >= args.val_start and (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_loss_dic, val_acc_dic = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                loss_func=loss_func,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "loss: {:.4f}".format(val_loss_dic["loss"]),
                    "self loss: {:.4f}".format(val_loss_dic["self_loss"]),
                    "mutual loss: {:.4f}".format(val_loss_dic["mutual_loss"]),
                    "dice_acc: {:.4f}".format(val_acc_dic["dice"]),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_loss", val_loss_dic["loss"], epoch)
                    writer.add_scalar("val_self_loss", val_loss_dic["self_loss"], epoch)
                    writer.add_scalar("val_mutual_loss", val_loss_dic["mutual_loss"], epoch)
                    writer.add_scalar("val_dice", val_acc_dic["dice"], epoch)

                if val_acc_dic["dice"] > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_acc_dic["dice"]))
                    val_acc_max = val_acc_dic["dice"]
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )

            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
