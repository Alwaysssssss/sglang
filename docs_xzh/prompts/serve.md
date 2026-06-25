本轮请你阅读/home/zhiheng/sglang/docs_xzh/hand_over/
  phase_e_current_status_and_e4_next_steps_handover.md这个交接文档和/home/
  zhiheng/sglang/docs_xzh/add_strategy，仔细了解项目背景。同时请你仔细分
  析/home/zhiheng/sglang_serve中/home/zhiheng/sglang_serve/python/sglang/
  multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py这个新加入的视
  频编辑模型的 pipeline 的 serve 服务部分，这里的服务是结合了公司的接口要
  求，对原 sglang 的 serve 部分做了一些修改的服务，我们下一步的工作暂时不
  推进加速工作，而是把这个服务的功能完全对齐接入到当前集成到 sglang 的
  Vivid-VR 模型。本轮只做代码分析，不修改代码。