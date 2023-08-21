import argparse
import collections
import random

parent = Path(__file__).parent.absolute()
sys.path.append(parent)
from iou_eval import *
from models_mae_image import *
from iopath.common.file_io import g_pathmgr as pathmgr


PARSER = argparse.ArgumentParser()
#
PARSER.add_argument('--cycles', type=int, default=5000, 
                    help='number of cycles')
PARSER.add_argument('--population_size', type=int, default=100, 
                    help='total population size')
PARSER.add_argument('--sample_size', type=int, default=25, 
                    help='sample size for selecting parent')
PARSER.add_argument('--video_encoder_depth', type=int, default=16, 
                    help='number of video encoder blocks')
PARSER.add_argument('--video_decoder_depth', type=int, default=4, 
                    help='number of video decoder blocks')
PARSER.add_argument('--supernet_ckpt', type=str, 
                    help='checkpoint path for supernet')
PARSER.add_argument('--encoder_mutation_prob', type=float, default=0.80, 
                    help='probability of encoder mutation')


def load_model_search(model_without_ddp, supernet_ckpt):
    with pathmgr.open(supernet_ckpt, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)

    
def eval_arch(model):
#   Args:
#     model: model for evaluation

#   Returns:
#     single_mean_2x2: the single_mean_2x2 evaluated on the davis videos

    store_path = '/shared/dannyt123/video_inpainting/video_mae_code/mae_image/davis_segs_search'
    single_prompt_csv = '/shared/dannyt123/video_inpainting/video_mae_code/datasets/davis_single_prompt.csv'
    prompt_csv = '/shared/dannyt123/video_inpainting/video_mae_code/datasets/davis_prompt.csv'
    davis_prompt_path = '/shared/dannyt123/video_inpainting/test_videos/davis_prompt'
    davis_2x2_prompt_path = '/shared/dannyt123/video_inpainting/test_videos/davis_2x2_single_prompt'
    davis_image_prompt_path = '/shared/dannyt123/video_inpainting/test_images/davis_image_prompts'
    generate_segmentations(model, store_path, single_prompt_csv, prompt_csv, davis_prompt_path, davis_2x2_prompt_path, davis_image_prompt_path, mae_image=True)
    single_mean_2x2, _ = run_evaluation_method(store_path)
    return single_mean_2x2
    
    
def mutate_arch(model_arch, encoder_all_indices, decoder_all_indices, encoder_mutation_prob):
#   Args:
#     model_arch: model architecture
#     encoder_all_indices: all the possible encoder indices
#     decoder_all_indices: all the possible decoder indices
#     encoder_mutation_prob: all the possible decoder indices

#   Returns:
#     video_encoder_indices: mutated video_encoder_indices with probability encoder_mutation_prob
#     video_decoder_indices: mutated video_decoder_indices with probability (1 - encoder_mutation_prob)

    video_encoder_indices, video_decoder_indices, _ = model_arch
    print('video_encoder_indices before: ', video_encoder_indices)
    print('video_decoder_indices before: ', video_decoder_indices)
    if random.randrange(100) < (encoder_mutation_prob * 100):
        video_encoder_indices.remove(random.choice(video_encoder_indices))
        diff_video_encoder_indices = list(set(encoder_all_indices) - set(video_encoder_indices))
        video_encoder_indices.append(random.choice(diff_video_encoder_indices))
    else:
        video_decoder_indices.remove(random.choice(video_decoder_indices))
        diff_video_decoder_indices = list(set(decoder_all_indices) - set(video_decoder_indices))
        video_decoder_indices.append(random.choice(diff_video_decoder_indices))
        
    print('video_encoder_indices after: ', video_encoder_indices)
    print('video_decoder_indices after: ', video_decoder_indices)
    return video_encoder_indices, video_decoder_indices


def regularized_evolution(cycles, population_size, sample_size, video_encoder_depth, video_decoder_depth, supernet_ckpt, encoder_mutation_prob=0.80):
#   """Algorithm for regularized evolution (i.e. aging evolution).
  
#   Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
#   Classifier Architecture Search".
  
#   Args:
#     cycles: the number of cycles the algorithm should run for.
#     population_size: the number of individuals to keep in the population.
#     sample_size: the number of individuals that should participate in each
#         tournament.

#   Returns:
#     history: a list of `Model` instances, representing all the models computed
#         during the evolution experiment.
#   """
    #Random video indices init
    model = mae_model_search(video_encoder_depth=video_encoder_depth, video_decoder_depth=video_decoder_depth, 
                                 video_encoder_indices=[i for i in range(video_encoder_depth)], video_decoder_indices=[i for i in range(video_decoder_depth)])
    load_model_search(model, supernet_ckpt)
    model.to('cuda')
    
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    depth = 24
    decoder_depth = 8
    encoder_all_indices = [i for i in range(depth + video_encoder_depth)]
    decoder_all_indices = [i for i in range(decoder_depth + video_decoder_depth)]
    while len(population) < population_size:
        video_encoder_indices = random.sample(encoder_all_indices, video_encoder_depth)
        video_decoder_indices = random.sample(decoder_all_indices, video_decoder_depth)
        
        model.video_encoder_indices = video_encoder_indices
        model.video_decoder_indices = video_decoder_indices
        single_mean_2x2 = eval_arch(model)
        print('[video_encoder_indices, video_decoder_indices, single_mean_2x2]: ', [video_encoder_indices, video_decoder_indices, single_mean_2x2])
        population.append([video_encoder_indices, video_decoder_indices, single_mean_2x2])
        history.append([video_encoder_indices, video_decoder_indices, single_mean_2x2])

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = random.choices(population, k=sample_size)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i[2])
        print('parent: ', parent)
            
        # Create the child model and store it.
        child_video_encoder_indices, child_video_decoder_indices = mutate_arch(parent, encoder_all_indices, decoder_all_indices, encoder_mutation_prob)
        model.video_encoder_indices = child_video_encoder_indices
        model.video_decoder_indices = child_video_decoder_indices
        
        single_mean_2x2 = eval_arch(model)
        print('[child_video_encoder_indices, child_video_decoder_indices, single_mean_2x2]: ', [child_video_encoder_indices, child_video_decoder_indices, single_mean_2x2])
        population.append([child_video_encoder_indices, child_video_decoder_indices, single_mean_2x2])
        history.append([child_video_encoder_indices, child_video_decoder_indices, single_mean_2x2])

        # Remove the oldest model.
        population.popleft()

    return history


def main():
    args = PARSER.parse_known_args()[0]
    regularized_evolution(args.cycles, args.population_size, args.sample_size, args.video_encoder_depth, args.video_decoder_depth, args.supernet_ckpt, args.encoder_mutation_prob)
    
if __name__ == "__main__":
    main()