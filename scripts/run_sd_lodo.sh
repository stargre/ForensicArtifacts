#!/usr/bin/env bash
# ============================================================
# SD + LODO 8 experiments launcher
# 运行一次 run.sh，自动完成：
#   SD   : single_AIGC / single_doc / single_df / single_imdl
#   LODO : without_AIGC / without_doc / without_df / without_imdl
#
# 协议：
#   SD:
#     train = 单个 domain
#     val   = 单个 domain
#     test  = null，全量 test_large.json
#
#   LODO:
#     train = 除 held-out domain 外的其余 domain
#     val   = 除 held-out domain 外的其余 domain
#     test  = null，全量 test_large.json
#
# 注意：
#   实验名中的 df 对应 JSON 里的真实 domain = deepfake
#   实验名中的 doc 对应 JSON 里的真实 domain = Doc
# ============================================================

set -euo pipefail

# ==================== 基础配置 ====================
yaml_config="${yaml_config:-/mnt/data3/zhiyu/ForensicArtifacts/config/lora/dino_lora_baseline.yaml}"
script_path="${script_path:-train-2.py}"

if [ ! -f "$yaml_config" ]; then
    echo "❌ 基础配置文件不存在: $yaml_config"
    exit 1
fi

if [ ! -f "$script_path" ]; then
    echo "❌ 训练脚本不存在: $script_path"
    echo "   请先执行: cp train-ema.py train-2.py"
    exit 1
fi

# ==================== 从基础 YAML 读取参数 ====================
gpus=$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$yaml_config", "r", encoding="utf-8"))
print(cfg["gpus"])
PY
)

base_flag=$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$yaml_config", "r", encoding="utf-8"))
print(cfg.get("flag", "train"))
PY
)

base_log_dir=$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$yaml_config", "r", encoding="utf-8"))
print(cfg.get("log_dir", "./log"))
PY
)

base_save_dir=$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$yaml_config", "r", encoding="utf-8"))
print(cfg.get("save_dir", "./checkpoints"))
PY
)

if [ "$base_flag" != "train" ]; then
    echo "❌ SD/LODO 批量实验要求基础配置 flag=train，当前 flag=$base_flag"
    exit 1
fi

gpu_count=$(echo "$gpus" | awk -F',' '{print NF}')

# ==================== 实验总目录 ====================
exp_root="${exp_root:-${base_save_dir}_SD_LODO}"
config_root="${exp_root}/generated_configs"
summary_csv="${exp_root}/sd_lodo_summary.csv"

mkdir -p "$exp_root"
mkdir -p "$config_root"

export TMPDIR="${TMPDIR:-/mnt/data2/zhiyu/tmp}"
mkdir -p "$TMPDIR"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$gpus"

echo "=========================================="
echo "  SD + LODO 8 实验启动"
echo "=========================================="
echo "  基础配置: $yaml_config"
echo "  训练脚本: $script_path"
echo "  使用 GPU: $gpus，共 $gpu_count 张"
echo "  输出根目录: $exp_root"
echo "  配置生成目录: $config_root"
echo "=========================================="

# ==================== domain 定义 ====================
# exp name 用于目录命名；domain value 必须和 JSON 里的 domain 字段完全一致
EXP_NAMES=("AIGC" "doc" "df" "imdl")
DOMAIN_VALUES=("AIGC" "Doc" "deepfake" "IMDL")

# ==================== 生成单个实验 YAML 的函数 ====================
generate_config() {
    local exp_name="$1"
    local mode="$2"
    local train_domains_csv="$3"
    local val_domains_csv="$4"
    local out_yaml="$5"
    local exp_save_dir="$6"
    local exp_log_dir="$7"

    python3 - <<PY
import copy
import json
import os
import yaml
from collections import Counter

base_yaml = "$yaml_config"
out_yaml = "$out_yaml"
exp_name = "$exp_name"
mode = "$mode"
train_domains_csv = "$train_domains_csv"
val_domains_csv = "$val_domains_csv"
exp_save_dir = "$exp_save_dir"
exp_log_dir = "$exp_log_dir"

def split_csv(x):
    if x is None or str(x).strip() == "":
        return None
    return [v.strip() for v in str(x).split(",") if v.strip()]

def load_json_domains(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    domains = [s.get("domain", "Unknown") for s in data if isinstance(s, dict)]
    return Counter(domains)

def assert_domains_exist(json_path, domains, split_name):
    if domains is None:
        return
    cnt = load_json_domains(json_path)
    missing = [d for d in domains if cnt.get(d, 0) == 0]
    if missing:
        raise ValueError(
            f"[{exp_name}] {split_name} target_domains={domains} 中存在无样本 domain: {missing}\\n"
            f"JSON: {json_path}\\n"
            f"该 JSON 中的 domain 分布: {dict(cnt)}"
        )

with open(base_yaml, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

train_domains = split_csv(train_domains_csv)
val_domains = split_csv(val_domains_csv)

# 只改 train / val；test 永远不筛选，统一全量 test_large.json
cfg["train_dataset"]["target_domains"] = train_domains
cfg["val_dataset"]["target_domains"] = val_domains
cfg["test_datasets"]["target_domains"] = None

# 其他筛选条件保持基础配置原样，不主动改 target_labels / target_mani_types
# 但如果你的基础配置里本来有 target_labels / target_mani_types，会继续生效。

cfg["flag"] = "train"
cfg["checkpoint_path"] = None
cfg["resume"] = False
cfg["save_dir"] = exp_save_dir
cfg["log_dir"] = exp_log_dir

# 写入实验信息，方便之后追踪
cfg["experiment"] = {
    "name": exp_name,
    "mode": mode,
    "train_domains": train_domains,
    "val_domains": val_domains,
    "test_domains": None,
    "test_policy": "all_domains_null",
}

# 训练前检查，避免跑到 Dataset 才报“筛选后无有效样本”
assert_domains_exist(cfg["train_dataset"]["path"], train_domains, "train")
assert_domains_exist(cfg["val_dataset"]["path"], val_domains, "val")

# test 为 None，不检查 target_domains；默认全量测试
os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
with open(out_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print(f"[生成配置] {exp_name}")
print(f"  mode : {mode}")
print(f"  train: {train_domains}")
print(f"  val  : {val_domains}")
print(f"  test : None，完整 test_large.json")
print(f"  yaml : {out_yaml}")
PY
}

# ==================== 运行单个实验 ====================
run_one_experiment() {
    local exp_name="$1"
    local mode="$2"
    local train_domains_csv="$3"
    local val_domains_csv="$4"

    local exp_save_dir="${exp_root}/${exp_name}"
    local exp_log_dir="${base_log_dir}_SD_LODO/${exp_name}"
    local exp_yaml="${config_root}/${exp_name}.yaml"

    mkdir -p "$exp_save_dir"
    mkdir -p "$exp_log_dir"

    generate_config "$exp_name" "$mode" "$train_domains_csv" "$val_domains_csv" "$exp_yaml" "$exp_save_dir" "$exp_log_dir"

    echo ""
    echo "============================================================"
    echo "▶ 开始实验: $exp_name"
    echo "  mode : $mode"
    echo "  train: $train_domains_csv"
    echo "  val  : $val_domains_csv"
    echo "  test : ALL(null)"
    echo "  log  : $exp_log_dir/train.log"
    echo "============================================================"

    if [ "$gpu_count" -gt 1 ]; then
        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node="${gpu_count}" \
            "$script_path" \
            --config "$exp_yaml" \
            2> "${exp_log_dir}/error.log" \
            1> "${exp_log_dir}/train.log"
    else
        python3 "$script_path" \
            --config "$exp_yaml" \
            2> "${exp_log_dir}/error.log" \
            1> "${exp_log_dir}/train.log"
    fi

    echo "✅ 实验完成: $exp_name"
}

# ==================== 1) SD: 单域训练/验证，全域测试 ====================
for i in "${!EXP_NAMES[@]}"; do
    short="${EXP_NAMES[$i]}"
    domain="${DOMAIN_VALUES[$i]}"
    exp_name="single_${short}"

    run_one_experiment "$exp_name" "SD" "$domain" "$domain"
done

# ==================== 2) LODO: 去一域训练/验证，全域测试 ====================
for i in "${!EXP_NAMES[@]}"; do
    heldout_short="${EXP_NAMES[$i]}"
    heldout_domain="${DOMAIN_VALUES[$i]}"
    exp_name="without_${heldout_short}"

    train_domains=()
    for j in "${!DOMAIN_VALUES[@]}"; do
        d="${DOMAIN_VALUES[$j]}"
        if [ "$d" != "$heldout_domain" ]; then
            train_domains+=("$d")
        fi
    done

    train_csv=$(IFS=','; echo "${train_domains[*]}")
    run_one_experiment "$exp_name" "LODO" "$train_csv" "$train_csv"
done

# ==================== 汇总结果 JSON 到 CSV，可选 ====================
echo ""
echo "============================================================"
echo "  尝试汇总 final_eval_summary.json"
echo "============================================================"

python3 - <<PY
import csv
import json
import os
from pathlib import Path

exp_root = Path("$exp_root")
summary_csv = Path("$summary_csv")

rows = []
for exp_dir in sorted(exp_root.iterdir()):
    if not exp_dir.is_dir() or exp_dir.name == "generated_configs":
        continue

    json_path = exp_dir / "final_eval_summary.json"
    row = {"experiment": exp_dir.name, "final_eval_summary_json": str(json_path)}

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))

            # 兼容之前建议的保存格式：
            # data["student"]["official_macro"] 或 data["student"]["test_mani_macro_with_val_thr"]["overall"]
            student = data.get("student", data)

            official = None
            if isinstance(student, dict):
                if "official_macro" in student:
                    official = student["official_macro"]
                elif "test_mani_macro_with_val_thr" in student:
                    official = student["test_mani_macro_with_val_thr"].get("overall")
                elif "test_metrics_with_val_thr" in student:
                    official = student["test_metrics_with_val_thr"]

            if official:
                for k in [
                    "accuracy", "balanced_accuracy", "auc_roc", "ap", "f1",
                    "precision", "recall", "specificity", "mcc", "kappa",
                    "threshold", "num_domains", "num_mani_types", "samples"
                ]:
                    row[f"official_{k}"] = official.get(k, "")
            else:
                row["note"] = "json_exists_but_metric_path_not_found"

        except Exception as e:
            row["note"] = f"json_parse_failed: {e}"
    else:
        row["note"] = "final_eval_summary.json not found; check train.log manually"

    rows.append(row)

all_keys = []
for r in rows:
    for k in r.keys():
        if k not in all_keys:
            all_keys.append(k)

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(rows)

print(f"汇总文件: {summary_csv}")
PY

echo ""
echo "=========================================="
echo "✅ 8 个实验全部完成"
echo "  输出根目录: $exp_root"
echo "  汇总 CSV : $summary_csv"
echo "=========================================="
