import torch
import numpy as np
import os
import time
from model.model import *
import timeit
import argparse
import utils.misc as misc
from dataset.getdataset import preparedataset
import json

if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9,10,11,12"

    parser = argparse.ArgumentParser(description='settings')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=150, help='epochs')
    parser.add_argument('--dataset', type=str, default="drugbank",
                        choices=["drugbank", "KIBA", "Davis"],
                        help='select dataset for training')
    parser.add_argument('--cuda', type=int, default=1, help='device,cuda:0,1,2...')
    parser.add_argument('--type', type=str, default="random", choices=["cold-protein","cold-smiles", "random"], help='cold or random')
    parser.add_argument('--usepl', type=str, default="False", choices=["True", "False"],
                        help='Use pseudo labeling or not')
    parser.add_argument('--output_file', type=str, default="random-result", help='output file name')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.cuda))
        print('The code uses GPU...')
    # else:1
    #     device = torch.device('cpu')
    #     print('The code uses CPU!!!')

    config = misc.load_config('./configs/random&cold.yaml')
    misc.seed_all(config.seed)

    config.lr.lr = float(config.lr.lr)
    config.lr.weight_decay = 1e-4
    config.lr.decay_interval = 5
    config.lr.lr_decay = 0.5

    # 存储所有fold的结果
    all_fold_results = []

    for fold in range(args.folds):
        print(f"\n{'=' * 50}")
        print(f"Starting fold {fold + 1}/{args.folds}")
        print(f"{'=' * 50}")

        endecoder = BiModalF(**config.settings, device=device)

        model = Predictor(endecoder, device)
        model.to(device)

        tranloader, validloader, testloader, _, _ = preparedataset(args.batch, args.type, args.dataset, fold)

        trainer = Trainer(model, config.lr.lr, config.lr.weight_decay, args.batch)
        tester = Tester(model)
        AUCs = ""

        # 创建按fold组织的输出目录
        dir_result = f'output/{args.dataset}/newresult/{args.output_file}/fold{fold}/'
        dir_model = f'output/{args.dataset}/newmodel/{args.output_file}/fold{fold}/'
        os.makedirs(dir_result, exist_ok=True)
        os.makedirs(dir_model, exist_ok=True)

        file_AUCs = os.path.join(dir_result, args.output_file + '.txt')
        file_model = os.path.join(dir_model, args.output_file + '.pt')

        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')

        print('Training...')
        print(AUCs)
        start = timeit.default_timer()

        max_AUC_dev = 0
        epoch_label = 0
        no_improvement_count = 0

        for epoch in range(1, args.epoch + 1):
            if epoch % config.lr.decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= config.lr.lr_decay

            if args.usepl == "False":
                loss_train = trainer.train(tranloader, device)
                AUC_dev, PRAUC_dev, AUPRC_dev, precision_dev, recall_dev, acc_dev, _ = tester.test(validloader, device)
                AUC_test, PRAUCtest, AUPRC_test, precision_test, recall_test, ACC_test, _ = tester.test(testloader,
                                                                                                        device)

                end = timeit.default_timer()
                time_elapsed = end - start

                AUCs = [epoch, time_elapsed, loss_train, AUC_dev, PRAUC_dev, precision_dev, recall_dev, acc_dev,
                        AUC_test, PRAUCtest, precision_test, recall_test, ACC_test]

                if AUC_dev > max_AUC_dev:
                    tester.save_model(model, file_model)
                    max_AUC_dev = AUC_dev
                    epoch_label = epoch
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                tester.save_AUCs(AUCs, file_AUCs)

                print(f"\nFold {fold} - Epoch {epoch}:")
                print("Validation:")
                print("AUC_dev\tPRAUC_dev\tprecision_dev\trecall_dev\tacc_dev")
                print(f"{AUC_dev:.4f}\t{PRAUC_dev:.4f}\t{precision_dev:.4f}\t{recall_dev:.4f}\t{acc_dev:.4f}")

                print("\nTest:")
                print("AUC_test\tPRAUCtest\tprecision_test\trecall_test\tACC_test")
                print(f"{AUC_test:.4f}\t{PRAUCtest:.4f}\t{precision_test:.4f}\t{recall_test:.4f}\t{ACC_test:.4f}")

                if no_improvement_count >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # 加载最佳模型并评估
        model.load_state_dict(torch.load(file_model))
        AUC_test, PRAUCtest, AUPRC_test, precision_test, recall_test, ACC_test, _ = tester.test(testloader, device)

        # 保存fold最终结果
        fold_results = {
            'fold': fold,
            'best_epoch': epoch_label,
            'AUC_test': AUC_test,
            'PRAUC': PRAUCtest,
            'ACC_test': ACC_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'F1': 2 * precision_test * recall_test / (precision_test + recall_test + 1e-5)
        }
        all_fold_results.append(fold_results)

        # 保存fold结果到文件
        results = 'Best Epoch\tAUC_test\tPRAUC\tAccuracy\tPrecision_test\tRecall_test\tF1'
        metric = [epoch_label, AUC_test, PRAUCtest, ACC_test,
                  precision_test, recall_test,
                  2 * precision_test * recall_test / (precision_test + recall_test + 1e-5)]

        with open(file_AUCs, 'a') as f:
            f.write(results + '\n')
            f.write('\t'.join(map(str, metric)) + '\n')

        print(f"Fold {fold} completed. Best epoch: {epoch_label}, Test AUC: {AUC_test:.4f}")

    # 计算并保存所有fold的平均结果
    print("\nFinal Cross-Validation Results:")
    avg_metrics = {
        'AUC_test': np.mean([r['AUC_test'] for r in all_fold_results]),
        'PRAUC': np.mean([r['PRAUC'] for r in all_fold_results]),
        'ACC_test': np.mean([r['ACC_test'] for r in all_fold_results]),
        'precision_test': np.mean([r['precision_test'] for r in all_fold_results]),
        'recall_test': np.mean([r['recall_test'] for r in all_fold_results]),
        'F1': np.mean([r['F1'] for r in all_fold_results])
    }

    print(f"Average AUC_test: {avg_metrics['AUC_test']:.4f}")
    print(f"Average PRAUC: {avg_metrics['PRAUC']:.4f}")
    print(f"Average ACC_test: {avg_metrics['ACC_test']:.4f}")
    print(f"Average Precision: {avg_metrics['precision_test']:.4f}")
    print(f"Average Recall: {avg_metrics['recall_test']:.4f}")
    print(f"Average F1: {avg_metrics['F1']:.4f}")

    # 保存平均结果
    final_dir = f'output/{args.dataset}/newresult/{args.output_file}/final/'
    os.makedirs(final_dir, exist_ok=True)
    final_file = os.path.join(final_dir, args.output_file + '_avg.txt')

    with open(final_file, 'w') as f:
        f.write("Cross-Validation Average Results:\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\nDetailed fold results:\n")
        for res in all_fold_results:
            f.write(
                f"Fold {res['fold']}: AUC={res['AUC_test']:.4f}, PRAUC={res['PRAUC']:.4f}, ACC={res['ACC_test']:.4f}\n")

    # 也可以保存为JSON
    with open(os.path.join(final_dir, args.output_file + '_avg.json'), 'w') as f:
        json.dump({
            'average': avg_metrics,
            'folds': all_fold_results
        }, f, indent=4)