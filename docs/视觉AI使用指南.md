# RDK X5视觉AI部署
RDK X5搭载计算单元为bpu整数运算单元，不支持yolo原生浮点模型的加速  
如果直接使用yolo模型在cpu上运行，大概率只有不到1的帧率  
因此我们需要转换yolo模型为.bin模型以便在bpu上运行，可以使用官方教程一步步转换  
不过官方教程太过麻烦，建议使用大佬做好的可视化转换工具https://github.com/xiongqi123123/RDK_ToolChain  
先用docker下载部署，然后按照步骤转换导出bin模型即可  
bin模型使用需要自己写处理，有官方的代码，我修改后的视觉识别代码放在camera文件夹内，可自行查阅  
