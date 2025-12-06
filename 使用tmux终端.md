 # tmux终端
  1. 新开/重连 tmux 会话：tmux new -s entryrank（或 tmux attach -t entryrank）。
  2. 在 tmux 里跑脚本：HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= python scripts/run_entry_exit_rank_baostock.py
  3. 断开但不结束：按 Ctrl+b 然后按 d（detach）。
  4. 重新连上后恢复：tmux attach -t entryrank（忘了名字可 tmux ls）。

# 硬件资源占用
- 实时 top（含线程）：top -H -p $(pgrep -d, -f run_entry_exit_rank_baostock.py)；退出按 q。

   1) 总览：CPU/内存/IO/负载/进程Top
  htop

   2) 仅CPU与IO详细（含每核使用率、IO吞吐）
  atop 1

   3) 只看本脚本的CPU/内存
  watch -n 2 "ps -u $USER -o pid,etime,%cpu,%mem,cmd | grep run_entry_exit_rank_baostock.py | grep -v grep"

   4) 简单总览（非交互）
  watch -n 2 "uptime && free -h && vmstat 1 2 | tail -1"

HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= PYTHONUNBUFFERED=1 \
nice -n 5 /home/ubuntu/.conda/envs/quant_env/bin/python scripts/run_entry_exit_rank_baostock.py --recompute \
| tee logs/entry_exit_rank_baostock_$(date +%Y%m%d%H%M).log

mkdir -p logs && \
  HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= PYTHONUNBUFFERED=1 \
  nice -n 5 /home/ubuntu/.conda/envs/quant_env/bin/python scripts/run_entry_exit_rank_baostock.py
  --recompute \
  | tee logs/entry_exit_rank_baostock_$(date +%Y%m%d%H%M).log

mkdir -p logs && \
  HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= PYTHONUNBUFFERED=1 \
  nice -n 5 /home/ubuntu/.conda/envs/quant_env/bin/python scripts/run_entry_exit_rank_baostock.py --recompute --workers 8 \
  | tee logs/entry_exit_rank_baostock_$(date +%Y%m%d%H%M).log

  mkdir -p logs && \
  HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= PYTHONUNBUFFERED=1 \
  nice -n 5 /home/ubuntu/.conda/envs/quant_env/bin/python scripts/run_entry_exit_rank_baostock.py --recompute --workers 6 \
  | tee logs/entry_exit_rank_baostock_$(date +%Y%m%d%H%M).log