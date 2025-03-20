from transformers import AutoConfig
import types

def patch_mert_config():
    """
    修补MERTConfig类，添加缺少的conv_pos_batch_norm属性
    在加载MERT模型之前调用此函数
    """
    # 获取MERT配置类
    mert_config = AutoConfig.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    
    # 检查是否已经有这个属性
    if not hasattr(mert_config.__class__, 'conv_pos_batch_norm'):
        # 添加缺少的属性
        setattr(mert_config.__class__, 'conv_pos_batch_norm', property(lambda self: True))
        print("已成功修补MERTConfig类，添加了conv_pos_batch_norm属性")
    
    return mert_config.__class__ 