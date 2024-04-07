import torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def create_punctuation_pipeline():
    inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='/mgData3/yangbo/sherpa-onnx/checkpoints/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727',
    model_revision="v2.0.4")
    return inference_pipeline


def create_vad_pipeline():
    inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='/mgData3/yangbo/sherpa-onnx/checkpoints/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    model_revision="v2.0.4")
    return inference_pipeline


class Server():
    def __init__(self):
        self.vad_model = create_vad_pipeline()
        self.punc_model = create_punctuation_pipeline()
        
    def perform_vad(self, wav):
        return self.vad_model(wav)          

    def perform_punc(self, result, cache):
        print(f"cache before: {cache}")
        punc_result = self.punc_model(result, cache=cache)[0]['text']
        print(f"cache after: {cache}")
        return punc_result


def deep_update(original, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            deep_update(original[key], value)
        else:
            original[key] = value


def test(**cfg):
    # deep_update(kwargs, cfg)
    kwargs.update(cfg)
    test1(**kwargs)
    print(f"cfg: {cfg}")
    print(f"kwargs: {kwargs}")


def test1(cache={}):
    cache['pre_text'] = [1, 2, 3]
    return
        

if __name__ == '__main__':
    wav = "/mgData3/yangbo/sherpa-onnx/checkpoints/zh_asr_zipformer_scale_280M_20240320/zh_voice_1.wav"
    inputs = "跨境河流是养育沿岸|人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员|在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险|向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切|愿意进一步完善双方联合工作机制|凡是|中方能做的我们|都会去做而且会做得更好我请印度朋友们放心中国在上游的|任何开发利用都会经过科学|规划和论证兼顾上下游的利益"
    inputs1 = "数据显示消费已连续五年成为拉动中国经济增长的第一动力|随着增加居民收入改善消费环境提升产品质量等一系列政策措施加快推进中国消费潜力将进一步释放推动人们的美好生活持续改善"
    inputs2 = "数据显示消费已连续五年成为拉动中国经济增长的第一动力"
    server = Server()
    for _ in range(2):
        cache = {'pre_text': []}
        vads = inputs1.split("|")
        for vad in vads:
            print(server.perform_punc(vad, cache=cache))
    # kwargs = {}
    # test(cache={'pre_text': []})
    # test(cache={'pre_text': []})
    
    
# for vad in vads:
#     rec_result = inference_pipeline(vad, cache=cache)
#     rec_result_all += rec_result[0]['text']

# print(rec_result_all)