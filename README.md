
### 关于

通过标准的 OpenAI API 格式访问大模型。

### 特点
1. 支持多种大模型，易扩展。
    - [x] OpenAI ChatGPT 系列模型（支持 Azure OpenAI API）
    - [x] Anthropic Claude 系列模型
    - [x] Google Gemini 系列模型
    - [x] 百度文心一言系列模型
    - [x] 阿里通义千问系列模型
    - [x] 讯飞星火认知大模型
    - [x] 智谱 ChatGLM 系列模型
    - [x] Moonshot AI
    - [x] 百川大模型
    - [x] ai360模型
    - [x] 字节云雀模型
2. 支持流式和非流式请求。
3. 支持key限流。
4. 管理key接口。
5. 异步日志收集。

### 快速开始

#### Docker 启动
```shell
docker run -d -p 8000:8000 \
   --name one-key kewei159/one-key:latest
```

#### Docker compose 启动
```shell
mkdir one-key
cd one-key
curl -O https://raw.githubusercontent.com/kyle-kw/one-key/main/docker-compose.yml
docker-compose up -d
```

#### 本地启动

建议使用conda创建虚拟环境。其他管理虚拟环境工具也可以。
```shell
conda create -y python=3.9 -n one-key
conda activate one-key
```

拉取代码并安装依赖
```shell
git clone https://github.com/kyle-kw/one-key.git
cd one-key
pip install -r requirements.txt
```

启动服务
```shell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```


