# <p align=center>Triboulet</p>

<p align="center">
<img align="center" src="https://github.com/user-attachments/assets/c027171a-5c4f-4264-9981-06adc6e8dc8b">
</p>

Making a full process from data collection to GRPO for a decent Mini model

Features/roadmap:

 - [ ] Webscraping, HF downloads, Annas Archive, etc to text files in a folder
 - [ ] Multiple tokenizers, chars, BPE
 - [ ] Sharded Data loader, txt files to config sizeable bin shards
 - [ ] Modeled out LM arch, Base Transformer, DS-MoE, MLA+NSA, RoPE, KVcache
 - [ ] Fully distributed Training Runs (DDP/FSDP/TP)

Deprecated Features (were tested, removed):

 - MTP: It needs scale to work, probably need 7b+
 - Mixture of a Million experts and Ultra Mem: I liked them, but its oddities and other performance issues force me to stay away, may keep just MoM though, more robust

Citations:

```
@misc{vaswani2023attentionneed,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1706.03762}, 
}


```

