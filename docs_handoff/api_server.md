

curl --location '127.0.0.1:5402/v1/videos/repairs' \
--header 'Content-Type: application/json' \
--data '{
        "bbox_expand_scale": 0.3,
        "dilate_px": 8,
        "feather_px": 8,
        "maskUrl": "flowcut/test/2026/07/16/d7395086afc2116bb87b0c11400836142d903d81.json",
        "mask_scale": 1,
        "num_inference_steps": 40,
        "prompt": "头发换成橙红色，自然点，\n视频画面自然流畅，替换后的元素与原视频运动、光影及镜头移动完全同步一致，时序稳定无闪烁。",
        "referenceImageUrl": "flowcut/test/2026/07/16/fcbd87e6624ece681c1d863d9a89957a8ef717de.png",
        "seed": 1643684109,
        "type": "objectedit",
        "videoUrl": "flowcut/test/2026/07/16/b5d7c82008aecc1ee5f1d3ac816473b1d9cb1741.mov",
        "minioConfig": {
            "access_key": "",
            "bucket_name": "vrs-mms-1258229344",
            "endpoint": "cos.ap-beijing.myqcloud.com",
            "prefix": "/flowcut",
            "provider": "cos",
            "region": "ap-beijing",
            "secret_key": "",
            "secure": true
        }
    }'

https://cos.ap-beijing.myqcloud.com/vrs-mms-1258229344/flowcut/2026/09/25/075417_videoedit-bb50e3b46e934d5bb507de6e839bf2d1.mov