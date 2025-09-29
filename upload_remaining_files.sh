#!/bin/bash

# 后台上传脚本 - 分批添加并推送所有剩余文件
# 排除指定的文件

LOG_FILE="/research/cbim/vast/cj574/wujiang_code/epo/upload.log"
REPO_DIR="/research/cbim/vast/cj574/wujiang_code/epo"

# 排除的文件列表
EXCLUDED_FILES=(
    "examples/general_running_local.sh"
    "instruct_to_run_alfworld.sh"
    "instruct_to_run_sciworld.sh"
    "instruct_to_run_webshop.sh"
    "draw_comparison.py"
)

echo "开始上传剩余文件 - $(date)" > "$LOG_FILE"

cd "$REPO_DIR"

# 清理可能的锁文件
rm -f .git/index.lock

# 获取所有未追踪的文件和目录
echo "检查未追踪的文件..." >> "$LOG_FILE"
untracked_items=($(git status --porcelain | grep '^??' | cut -c4-))

# 分批添加文件
batch_size=5
total_items=${#untracked_items[@]}
echo "共发现 $total_items 个未追踪的项目" >> "$LOG_FILE"

for ((i=0; i<total_items; i+=batch_size)); do
    batch_end=$((i+batch_size-1))
    if [ $batch_end -ge $total_items ]; then
        batch_end=$((total_items-1))
    fi
    
    echo "处理批次 $((i/batch_size+1))：项目 $((i+1))-$((batch_end+1))" >> "$LOG_FILE"
    
    # 添加当前批次的项目
    for ((j=i; j<=batch_end; j++)); do
        item="${untracked_items[j]}"
        echo "  添加: $item" >> "$LOG_FILE"
        
        # 检查是否在排除列表中
        skip=false
        for excluded in "${EXCLUDED_FILES[@]}"; do
            if [[ "$item" == "$excluded" ]]; then
                echo "  跳过排除文件: $item" >> "$LOG_FILE"
                skip=true
                break
            fi
        done
        
        if [ "$skip" = false ]; then
            git add "$item" 2>> "$LOG_FILE"
            if [ $? -eq 0 ]; then
                echo "  成功添加: $item" >> "$LOG_FILE"
            else
                echo "  添加失败: $item" >> "$LOG_FILE"
            fi
        fi
    done
    
    # 每处理几个批次后提交一次
    if [ $((i/batch_size % 3)) -eq 2 ] || [ $batch_end -eq $((total_items-1)) ]; then
        echo "创建提交..." >> "$LOG_FILE"
        commit_msg="Add remaining EPO files - batch $((i/batch_size+1))"
        
        git commit -m "$commit_msg" >> "$LOG_FILE" 2>&1
        if [ $? -eq 0 ]; then
            echo "提交成功" >> "$LOG_FILE"
            
            # 推送到远程
            echo "推送到远程仓库..." >> "$LOG_FILE"
            git push origin main >> "$LOG_FILE" 2>&1
            if [ $? -eq 0 ]; then
                echo "推送成功" >> "$LOG_FILE"
            else
                echo "推送失败，稍后重试" >> "$LOG_FILE"
            fi
        else
            echo "提交失败或无内容可提交" >> "$LOG_FILE"
        fi
    fi
    
    # 短暂休息避免系统负载过高
    sleep 2
done

# 最终检查并推送任何剩余的更改
echo "最终检查..." >> "$LOG_FILE"
if ! git diff --cached --quiet; then
    echo "发现剩余暂存文件，创建最终提交" >> "$LOG_FILE"
    git commit -m "Add final remaining EPO files

Complete upload of all EPO project files excluding local scripts" >> "$LOG_FILE" 2>&1
    
    git push origin main >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        echo "最终推送成功" >> "$LOG_FILE"
    else
        echo "最终推送失败" >> "$LOG_FILE"
    fi
fi

# 移除排除的文件（如果它们被意外添加）
echo "清理排除的文件..." >> "$LOG_FILE"
for excluded in "${EXCLUDED_FILES[@]}"; do
    if git ls-files --error-unmatch "$excluded" >/dev/null 2>&1; then
        echo "移除意外添加的排除文件: $excluded" >> "$LOG_FILE"
        git rm --cached "$excluded" >> "$LOG_FILE" 2>&1
    fi
done

# 如果有移除操作，再次提交
if ! git diff --cached --quiet; then
    git commit -m "Remove excluded files from repository" >> "$LOG_FILE" 2>&1
    git push origin main >> "$LOG_FILE" 2>&1
fi

echo "上传完成 - $(date)" >> "$LOG_FILE"
echo "============================================" >> "$LOG_FILE"