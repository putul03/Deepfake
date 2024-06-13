import ast
import pprint

class HParams(object):
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def __setitem__(self, key, value): setattr(self, key, value)
    def __getitem__(self, key): return getattr(self, key)
    def __repr__(self): return pprint.pformat(self.__dict__)

    def parse(self, string):
        # Overrides hparams from a comma-separated string of name=value pairs
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self

hparams = HParams(
        ### Signal Processing (used in both synthesizer and vocoder)
        sample_rate = 16000,
        n_fft = 800,
        num_mels = 80,
        hop_size = 200,                             
        win_size = 800,                            
        fmin = 55,
        min_level_db = -100,
        ref_level_db = 20,
        max_abs_value = 4.,                     
        preemphasis = 0.97,                    
        preemphasize = True,

        tts_embed_dims = 512,                      
        tts_encoder_dims = 256,
        tts_decoder_dims = 128,
        tts_postnet_dims = 512,
        tts_encoder_K = 5,
        tts_lstm_dims = 1024,
        tts_postnet_K = 5,
        tts_num_highways = 4,
        tts_dropout = 0.5,
        tts_cleaner_names = ["english_cleaners"],
        tts_stop_threshold = -3.4,                 
        tts_schedule = [(2,  1e-3,  20_000,  12),   
                        (2,  5e-4,  40_000,  12),   
                        (2,  2e-4,  80_000,  12),   
                        (2,  1e-4, 160_000,  12),   
                        (2,  3e-5, 320_000,  12),   
                        (2,  1e-5, 640_000,  12)],  

        tts_clip_grad_norm = 1.0,                   
        tts_eval_interval = 500,                   
        tts_eval_num_samples = 1,        

        ### Data Preprocessing
        max_mel_frames = 900,
        rescale = True,
        rescaling_max = 0.9,
        synthesis_batch_size = 16,                  

        ### Mel Visualization and Griffin-Lim
        signal_normalization = True,
        power = 1.5,
        griffin_lim_iters = 60,

        ### Audio processing options
        fmax = 7600,                                
        allow_clipping_in_normalization = True,     
        clip_mels_length = True,                    
        use_lws = False,                            
        symmetric_mels = True,                      
                                                   
        trim_silence = True,              

        ### SV2TTS
        speaker_embedding_size = 256,              
        silence_min_duration_split = 0.4,           
        utterance_min_duration = 1.6,               
        )

def hparams_debug_string():
    return str(hparams)
