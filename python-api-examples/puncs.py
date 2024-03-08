from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='/mgData3/yangbo/sherpa-onnx/checkpoints/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727',
    model_revision="v2.0.4")

inputs = "跨境河流是养育沿岸|人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员|在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险|向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切|愿意进一步完善双方联合工作机制|凡是|中方能做的我们|都会去做而且会做得更好我请印度朋友们放心中国在上游的|任何开发利用都会经过科学|规划和论证兼顾上下游的利益"
vads = inputs.split("|")

rec_result_all="outputs: "
cache = {}
for vad in vads:
    rec_result = inference_pipeline(vad, cache=cache)
    rec_result_all += rec_result[0]['text']

print(rec_result_all)