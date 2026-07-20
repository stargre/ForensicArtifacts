#!/usr/bin/env bash
# ============================================================
# Token Pooling Ablation Launcher
# 一次运行 8 个 pooling_type 实验
# ============================================================

set -euo pipefail

# ==================== 基础配置文件路径 ====================
yaml_config="${yaml_config:-/mnt/data3/zhiyu/ForensicArtifacts/config/lora/dino_lora_baseline.yaml}"

# 训练脚本：建议复制 train-ema.py 为 train-2.py，避免影响原文件
script_path="${TRAIN_SCRIPT:-train-ema.py}"

# 消融实验总名称
ablation_name="${ABLATION_NAME:-token_pooling_ablation}"

# ==================== 检查文件 ====================
if [ ! -f "$yaml_config" ]; then
    echo "❌ 配置文件不存在: $yaml_config"
    exit 1
fi

if [ ! -f "$script_path" ]; then
    echo "❌ 训练脚本不存在: $script_path"
    echo "   你可以先执行: cp train-ema.py train-ema.py"
    exit 1
fi

echo "=========================================="
echo "  Token Pooling Ablation"
echo "  基础配置文件: $yaml_config"
echo "  训练脚本: $script_path"
echo "=========================================="

# ==================== 解析基础 YAML ====================
gpus=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_config', encoding='utf-8'))['gpus'])")
flag=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_config', encoding='utf-8'))['flag'])")
base_log_dir=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_config', encoding='utf-8'))['log_dir'])")
base_save_dir=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_config', encoding='utf-8')).get('save_dir', './checkpoints'))")

gpu_count=$(echo "$gpus" | awk -F',' '{print NF}')

echo "  使用GPU: $gpus，共 $gpu_count 张"
echo "  运行模式: $flag"
echo "  基础日志目录: $base_log_dir"
echo "  基础权重目录: $base_save_dir"

if [ "$flag" != "train" ]; then
    echo "❌ 这个消融脚本只用于 train 模式，请把 YAML 中 flag 改为 train"
    exit 1
fi

# ==================== 环境设置 ====================
export TMPDIR=/mnt/data2/zhiyu/tmp
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${gpus}

mkdir -p "$TMPDIR"

# ==================== 8 个 pooling_type ====================
pooling_types=(
    "cls"
    "patch_mean"
    "patch_max"
    "cls_patch_mean"
    "cls_patch_max"
    "patch_mean_max"
    "cls_patch_mean_max"
    "cls_patch_attention_mean"
)

generated_config_dir="${base_log_dir}/${ablation_name}/generated_configs"
mkdir -p "$generated_config_dir"

summary_csv="${base_log_dir}/${ablation_name}/token_pooling_ablation_summary.csv"
mkdir -p "$(dirname "$summary_csv")"

echo ""
echo "=========================================="
echo "  即将运行以下 8 个实验"
echo "=========================================="
printf '  - %s\n' "${pooling_types[@]}"
echo "=========================================="
echo ""

# ============================================================
# 逐个实验运行
# ============================================================
for pooling in "${pooling_types[@]}"; do

    exp_config="${generated_config_dir}/${pooling}.yaml"

    echo ""
    echo "============================================================"
    echo "  开始实验: pooling_type=${pooling}"
    echo "============================================================"

    # --------------------------------------------------------
    # 生成当前实验专用 YAML
    # --------------------------------------------------------
    python3 - "$yaml_config" "$exp_config" "$pooling" "$base_log_dir" "$base_save_dir" "$ablation_name" <<'PY'
import os
import sys
import yaml
from pathlib import Path

base_yaml, out_yaml, pooling, base_log_dir, base_save_dir, ablation_name = sys.argv[1:]

with open(base_yaml, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["flag"] = "train"

# 每个 pooling 独立训练，不能从上一个实验 resume
cfg["resume"] = False
cfg["checkpoint_path"] = None

cfg.setdefault("model", {})
cfg["model"]["pooling_type"] = pooling

# 关键点：
# 只有 cls_patch_attention_mean 需要启用 attention_pooling
# 其他 pooling 必须关闭，避免 DDP 下 unused parameter
cfg["model"]["enable_attention_pooling"] = (pooling == "cls_patch_attention_mean")

# 每个实验单独目录，防止日志和 checkpoint 覆盖
cfg["log_dir"] = os.path.join(base_log_dir, ablation_name, pooling)
cfg["save_dir"] = os.path.join(base_save_dir, ablation_name, pooling)

Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)

with open(out_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        cfg,
        f,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False
    )

print(cfg["log_dir"])
PY

    exp_log_dir=$(python3 -c "import yaml; print(yaml.safe_load(open('$exp_config', encoding='utf-8'))['log_dir'])")
    exp_save_dir=$(python3 -c "import yaml; print(yaml.safe_load(open('$exp_config', encoding='utf-8'))['save_dir'])")

    mkdir -p "$exp_log_dir"
    mkdir -p "$exp_save_dir"

    echo "  当前配置: $exp_config"
    echo "  日志目录: $exp_log_dir"
    echo "  权重目录: $exp_save_dir"

    # --------------------------------------------------------
    # 启动训练
    # --------------------------------------------------------
    if [ "$gpu_count" -gt 1 ]; then
        echo "  使用 DDP 训练，GPU 数量: $gpu_count"

        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=${gpu_count} \
            ${script_path} \
            --config ${exp_config} \
            2> "${exp_log_dir}/error.log" \
            1> "${exp_log_dir}/train.log"
    else
        echo "  使用单 GPU 训练"

        python3 ${script_path} \
            --config ${exp_config} \
            2> "${exp_log_dir}/error.log" \
            1> "${exp_log_dir}/train.log"
    fi

    echo "✅ 实验完成: ${pooling}"
    echo "  train.log: ${exp_log_dir}/train.log"
    echo "  error.log: ${exp_log_dir}/error.log"

done

# ============================================================
# 汇总 8 个实验最终指标
# 结果不依赖终端显示，统一写入 YAML log_dir 下的 summary 文件夹
# ============================================================

summary_dir="${base_log_dir}/${ablation_name}/summary"
mkdir -p "${summary_dir}"

echo ""
echo "============================================================"
echo "  开始汇总 8 个实验结果"
echo "  汇总目录: ${summary_dir}"
echo "============================================================"

python3 - "$base_log_dir" "$ablation_name" "$summary_dir" <<'PY'
import os
import csv
import json
import sys

base_log_dir, ablation_name, summary_dir = sys.argv[1:]

pooling_types = [
    "cls",
    "patch_mean",
    "patch_max",
    "cls_patch_mean",
    "cls_patch_max",
    "patch_mean_max",
    "cls_patch_mean_max",
    "cls_patch_attention_mean",
]

summary_csv = os.path.join(summary_dir, "token_pooling_ablation_summary.csv")
summary_txt = os.path.join(summary_dir, "token_pooling_ablation_summary.txt")
summary_md = os.path.join(summary_dir, "token_pooling_ablation_summary.md")
summary_json = os.path.join(summary_dir, "token_pooling_ablation_summary.json")


def parse_metric_line(line):
    """
    解析类似：
    Final-Student | VAL-thr ManiType-Domain-Macro | ACC=xx% | BalACC=xx% | AUC=xx | AP=xx | F1=xx | P=xx | R=xx | ...
    
    注意：
    不能用简单正则 P=，否则会把 AP= 误解析成 Precision。
    所以这里按 | 分段后精确匹配 key。
    """
    key_map = {
        "ACC": "acc",
        "BalACC": "bal_acc",
        "AUC": "auc",
        "AP": "ap",
        "F1": "f1",
        "P": "precision",
        "R": "recall",
        "Spec": "specificity",
        "MCC": "mcc",
        "Kappa": "kappa",
        "Thr": "thr",
    }

    out = {
        "acc": "",
        "bal_acc": "",
        "auc": "",
        "ap": "",
        "f1": "",
        "precision": "",
        "recall": "",
        "specificity": "",
        "mcc": "",
        "kappa": "",
        "thr": "",
    }

    parts = [x.strip() for x in line.split("|")]

    for part in parts:
        if "=" not in part:
            continue

        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip().replace("%", "")

        if k in key_map:
            out[key_map[k]] = v

    return out


rows = []

for pooling in pooling_types:
    log_path = os.path.join(base_log_dir, ablation_name, pooling, "train.log")
    err_path = os.path.join(base_log_dir, ablation_name, pooling, "error.log")

    row = {
        "pooling_type": pooling,
        "status": "missing_log",

        # official: 使用 Val macro best threshold 的正式结果
        "official_acc": "",
        "official_bal_acc": "",
        "official_auc": "",
        "official_ap": "",
        "official_f1": "",
        "official_precision": "",
        "official_recall": "",
        "official_specificity": "",
        "official_mcc": "",
        "official_kappa": "",
        "official_thr": "",

        # oracle: Test 自己搜索 threshold，仅作为上界参考
        "oracle_acc": "",
        "oracle_bal_acc": "",
        "oracle_auc": "",
        "oracle_ap": "",
        "oracle_f1": "",
        "oracle_precision": "",
        "oracle_recall": "",
        "oracle_specificity": "",
        "oracle_mcc": "",
        "oracle_kappa": "",
        "oracle_thr": "",

        "log_path": log_path,
        "error_log_path": err_path,
    }

    if not os.path.exists(log_path):
        rows.append(row)
        continue

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    lines = text.splitlines()

    official_lines = [
        line for line in lines
        if "VAL-thr ManiType-Domain-Macro" in line
    ]

    oracle_lines = [
        line for line in lines
        if "TEST-thr ManiType-Domain-Macro" in line
    ]

    if official_lines:
        metrics = parse_metric_line(official_lines[-1])
        row.update({
            "official_acc": metrics["acc"],
            "official_bal_acc": metrics["bal_acc"],
            "official_auc": metrics["auc"],
            "official_ap": metrics["ap"],
            "official_f1": metrics["f1"],
            "official_precision": metrics["precision"],
            "official_recall": metrics["recall"],
            "official_specificity": metrics["specificity"],
            "official_mcc": metrics["mcc"],
            "official_kappa": metrics["kappa"],
            "official_thr": metrics["thr"],
        })
        row["status"] = "ok"
    else:
        row["status"] = "no_official_final_summary_found"

    if oracle_lines:
        metrics = parse_metric_line(oracle_lines[-1])
        row.update({
            "oracle_acc": metrics["acc"],
            "oracle_bal_acc": metrics["bal_acc"],
            "oracle_auc": metrics["auc"],
            "oracle_ap": metrics["ap"],
            "oracle_f1": metrics["f1"],
            "oracle_precision": metrics["precision"],
            "oracle_recall": metrics["recall"],
            "oracle_specificity": metrics["specificity"],
            "oracle_mcc": metrics["mcc"],
            "oracle_kappa": metrics["kappa"],
            "oracle_thr": metrics["thr"],
        })

    rows.append(row)


fieldnames = [
    "pooling_type",
    "status",

    "official_acc",
    "official_bal_acc",
    "official_auc",
    "official_ap",
    "official_f1",
    "official_precision",
    "official_recall",
    "official_specificity",
    "official_mcc",
    "official_kappa",
    "official_thr",

    "oracle_acc",
    "oracle_bal_acc",
    "oracle_auc",
    "oracle_ap",
    "oracle_f1",
    "oracle_precision",
    "oracle_recall",
    "oracle_specificity",
    "oracle_mcc",
    "oracle_kappa",
    "oracle_thr",

    "log_path",
    "error_log_path",
]

# 1. CSV：方便 Excel / pandas 读取
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# 2. JSON：方便后续脚本读取
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

# 3. TXT：方便直接 cat 查看
txt_lines = []
txt_lines.append("Token Pooling Ablation Summary")
txt_lines.append("=" * 140)
txt_lines.append(
    f"{'pooling_type':<28} "
    f"{'ACC':>8} "
    f"{'BalACC':>8} "
    f"{'AUC':>8} "
    f"{'AP':>8} "
    f"{'F1':>8} "
    f"{'MCC':>8} "
    f"{'Thr':>8} "
    f"{'status':>34}"
)
txt_lines.append("-" * 140)

for r in rows:
    txt_lines.append(
        f"{r['pooling_type']:<28} "
        f"{r['official_acc']:>8} "
        f"{r['official_bal_acc']:>8} "
        f"{r['official_auc']:>8} "
        f"{r['official_ap']:>8} "
        f"{r['official_f1']:>8} "
        f"{r['official_mcc']:>8} "
        f"{r['official_thr']:>8} "
        f"{r['status']:>34}"
    )

txt_lines.append("-" * 140)
txt_lines.append("")
txt_lines.append("说明：")
txt_lines.append("official_* 表示使用 Val macro best threshold 的正式测试结果。")
txt_lines.append("oracle_* 表示 Test 自己搜索 threshold 的上界参考结果，不建议作为正式对比指标。")
txt_lines.append("")
txt_lines.append(f"CSV : {summary_csv}")
txt_lines.append(f"JSON: {summary_json}")
txt_lines.append(f"MD  : {summary_md}")

with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(txt_lines))

# 4. Markdown：方便放论文、汇报、实验记录
md_lines = []
md_lines.append("# Token Pooling Ablation Summary")
md_lines.append("")
md_lines.append("正式结果使用 `VAL-thr ManiType-Domain-Macro`，即验证集按 `mani_type -> domain -> overall` 宏平均协议选阈值后，在测试集上评估。")
md_lines.append("")
md_lines.append("| Pooling Type | Status | ACC | BalACC | AUC | AP | F1 | Precision | Recall | Spec | MCC | Thr |")
md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in rows:
    md_lines.append(
        f"| {r['pooling_type']} "
        f"| {r['status']} "
        f"| {r['official_acc']} "
        f"| {r['official_bal_acc']} "
        f"| {r['official_auc']} "
        f"| {r['official_ap']} "
        f"| {r['official_f1']} "
        f"| {r['official_precision']} "
        f"| {r['official_recall']} "
        f"| {r['official_specificity']} "
        f"| {r['official_mcc']} "
        f"| {r['official_thr']} |"
    )

md_lines.append("")
md_lines.append("## Oracle / Upper Bound")
md_lines.append("")
md_lines.append("下面结果为 `TEST-thr ManiType-Domain-Macro`，即测试集自己搜索最佳阈值，只能作为上界参考。")
md_lines.append("")
md_lines.append("| Pooling Type | ACC | BalACC | AUC | AP | F1 | Precision | Recall | Spec | MCC | Thr |")
md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in rows:
    md_lines.append(
        f"| {r['pooling_type']} "
        f"| {r['oracle_acc']} "
        f"| {r['oracle_bal_acc']} "
        f"| {r['oracle_auc']} "
        f"| {r['oracle_ap']} "
        f"| {r['oracle_f1']} "
        f"| {r['oracle_precision']} "
        f"| {r['oracle_recall']} "
        f"| {r['oracle_specificity']} "
        f"| {r['oracle_mcc']} "
        f"| {r['oracle_thr']} |"
    )

md_lines.append("")
md_lines.append("## Log Paths")
md_lines.append("")
for r in rows:
    md_lines.append(f"- `{r['pooling_type']}`")
    md_lines.append(f"  - train log: `{r['log_path']}`")
    md_lines.append(f"  - error log: `{r['error_log_path']}`")

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

# 终端只输出文件位置，不再打印大表
print(f"Summary CSV : {summary_csv}")
print(f"Summary TXT : {summary_txt}")
print(f"Summary MD  : {summary_md}")
print(f"Summary JSON: {summary_json}")
PY

echo ""
echo "============================================================"
echo "✅ 8 个 Token Pooling 消融实验全部完成"
echo "  汇总目录: ${summary_dir}"
echo "  CSV : ${summary_dir}/token_pooling_ablation_summary.csv"
echo "  TXT : ${summary_dir}/token_pooling_ablation_summary.txt"
echo "  MD  : ${summary_dir}/token_pooling_ablation_summary.md"
echo "  JSON: ${summary_dir}/token_pooling_ablation_summary.json"
echo "============================================================"