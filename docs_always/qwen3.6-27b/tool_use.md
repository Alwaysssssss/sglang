使用 @docs_always/qwen3.6-27b/start_qwen36_27b.sh 启动服务，把API暴露给agent，但是agent无法调用工具，具体表现如下：

、、、
我来为你生成一个小女孩玩耍的图片。

<tool_call> <function=generate_image> <parameter=generation_type> text2image </parameter> <parameter=model_name> gpt-image-2 </parameter> <parameter=prompt_text> A cute little girl playing happily in a sunny park, wearing a colorful dress, laughing and running with a balloon, warm natural lighting, cheerful atmosphere, photorealistic style </parameter> </function> </tool_call>
、、、

可以生成工具调用的指令，但是无法真正调用工具，详解具体原因，先排查原因，是不是启动脚本哪里有问题，不要写任何代码，只排查原因