# Third-party notices

## Code in this repository

| where | origin | license | changes |
|---|---|---|---|
| Model backends, extraction, retrieval-head analysis, and bundled ranking assets | [parkquasar/Image-Retrieval-Heads](https://github.com/parkquasar/Image-Retrieval-Heads) | MIT | adapted for GAVR |
| `utils/conversation.py` | [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL), `internvl_chat/internvl/conversation.py`, itself modified from [lm-sys/FastChat](https://github.com/lm-sys/FastChat) | MIT and Apache-2.0 | header added |
| `utils/model_utils.py` `load_image_internvl`, `utils/model_backend.py` `_internvl_*` image helpers | InternVL demo code | MIT | adapted |
| `eval/eval_vhs_qwen3vl_embedder.py` `Qwen3VLForEmbedding` and pooling | [QwenLM/Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding), `src/models/qwen3_vl_embedding.py` | Apache-2.0 | modified |
| `eval/eval_vhs_qwen3vl_embedder.py` `_smart_resize_image` | qwen-vl-utils in [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | Apache-2.0 | reimplemented |
| `eval/eval_vhs_external_retrievers.py`, the 80 prompt templates | [openai/CLIP](https://github.com/openai/CLIP) | MIT | none |

The Apache License 2.0 is in `LICENSES/Apache-2.0.txt`. The MIT-licensed parts carry these notices:

```
InternVL: Copyright (c) 2023 OpenGVLab
CLIP:     Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Models and data

These are not included here and are used under their own terms.

- Models: InternVL3.5, Qwen3-VL, Qwen3-VL-Embedding and SigLIP (Apache-2.0), CLIP (MIT).
- Visual Haystacks (MIT).
- MS-COCO 2017: annotations under CC BY 4.0, images under the Flickr Terms of Use.
- ICIR-MCQ: see its [data card](https://huggingface.co/datasets/jerryray/ICIR-MCQ).
