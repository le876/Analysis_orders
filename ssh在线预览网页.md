- 端口转发 + HTTP 服务（最方便在线预览）
      1. 在 SSH 会话里进入仓库根目录，运行：
         python -m http.server 9000 --directory reports/visualization_analysis
      2. 在本地终端重新开一个窗口，连接时加端口转发：
         ssh -L 9000:localhost:9000 ubuntu@100.115.26.69
      3. 本地浏览器访问 http://localhost:9000/index.html 即可看到完整仪表板。结束后按 Ctrl+C 停止服务。