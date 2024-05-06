# CS570_Artificial_Intelligence
AI Project based on CS570 course work.

COLA, or Contrastive Learning for Audio, is a method tailored for audio data processing. This is a small scale implementation with FMA dataset for few epochs. It operates by selecting an anchor and a positive example from each audio clip, with the remaining samples in the batch serving as distractors. This approach enhances training difficulty, compelling the model to learn more informative representations. Additionally, the reuse of distractors from other samples within the batch minimizes the computational resources required for generating them.

Moreover, COLA employs a Bi-linear similarity measure, learned directly from the data, which significantly improves performance compared to traditional methods like Cosine similarity, resulting in an additional 7% average accuracy on downstream tasks.

In terms of evaluation, COLA demonstrates superiority over other techniques, such as triplet loss, achieving an extra 20% average accuracy across various tasks. This improvement is particularly notable when using the learned representations to train linear classifiers on tasks with labeled data.
