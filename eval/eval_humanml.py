from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from data_utils.humanml.utils.metrics import *
from data_utils.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_utils.humanml.scripts.motion_process import *
from data_utils.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model
import numpy as np
from scipy import linalg

from diffusion import logger
from utils import dist_util
from data_utils.get_data import get_dataset_loader

torch.multiprocessing.set_sharing_strategy('file_system')

def slice_motion_sample(sample, window_size, step_size=10):
    # sample [nframes, nfeats]
    windows = []
    max_offset = sample.shape[0] - window_size + 1
    for offset_i in np.arange(max_offset)[0::step_size]:
        windows.append(sample[offset_i:offset_i+window_size].unsqueeze(0))
    return torch.cat(windows, dim=0)



def get_gt_samples(args):
    try:
        return torch.tensor(np.load(args.sin_path), device=args.device).squeeze().permute(1, 0)
    except:
        return np.load(args.sin_path, allow_pickle=True)[None][0]['motion_raw'][0].to(args.device).squeeze().permute(1, 0)  # benchmark npy

def generate_eval_samples(model, diffusion, num_samples):
    sample_fn = diffusion.p_sample_loop
    model_kwargs = {'y': {}}
    sample = sample_fn(
        model,
        (num_samples, model.njoints, model.nfeats, 196),  # FIXME - 196?
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    sample = sample.squeeze().permute(0, 2, 1)

    return sample

def calc_inter_diversity(eval_wrapper, samples):
    motion_emb = eval_wrapper.get_motion_embeddings(samples, torch.tensor([samples.shape[1]] * samples.shape[0])).cpu().numpy()
    dist = linalg.norm(motion_emb[:samples.shape[0]//2] - motion_emb[samples.shape[0]//2:], axis=1)  # FIXME - will not work for odd bs
    return dist.mean()

def calc_sifid(eval_wrapper, gen_samples, gt_sample, window_size=40):

    gt_slices = slice_motion_sample(gt_sample, window_size)

    def get_stats(_samples):
        _motion_embeddings = eval_wrapper.get_motion_embeddings(_samples, torch.tensor([_samples.shape[1]] * _samples.shape[0])).cpu().numpy()
        _mu, _cov = calculate_activation_statistics(_motion_embeddings)
        return _mu, _cov

    sifid = []

    gt_mu, gt_cov = get_stats(gt_slices)

    for sampe_i in range(gen_samples.shape[0]):
        gen_slices = slice_motion_sample(gen_samples[sampe_i], window_size)
        gen_mu, gen_cov = get_stats(gen_slices)
        sifid.append(calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov))

    return np.array(sifid).mean()


def calc_intra_diversity(eval_wrapper, samples, window_size=40):
    max_offset = samples.shape[1] - window_size
    dist = []
    for sample_i in range(samples.shape[0]):
        offsets = np.random.randint(max_offset, size=2)
        window0 = samples[[sample_i], offsets[0]:offsets[0]+window_size, :]
        window1 = samples[[sample_i], offsets[1]:offsets[1]+window_size, :]
        motion_emb = eval_wrapper.get_motion_embeddings(torch.cat([window0, window1]), torch.tensor([window_size] * 2)).cpu().numpy()
        dist.append(linalg.norm(motion_emb[0] - motion_emb[1]))
    return np.array(dist).mean()


def evaluate(args, model, diffusion, eval_wrapper, num_samples_limit, replication_times):

    results = {}

    for window_size in [10, 20, 40]:
        print(f'===Starting [window_size={window_size}]===')
        intra_diversity = []
        gt_intra_diversity = []
        inter_diversity = []
        sifid = []
        for rep_i in range(replication_times):
            gt_samples = get_gt_samples(args)
            gen_samples = generate_eval_samples(model, diffusion, num_samples_limit)
            print(f'===REP[{rep_i}]===')
            _intra_diversity = calc_intra_diversity(eval_wrapper, gen_samples, window_size=window_size)
            intra_diversity.append(_intra_diversity)
            print('intra_diversity [{:.3f}]'.format(_intra_diversity))
            _gt_intra_diversity = calc_intra_diversity(eval_wrapper, torch.tile(gt_samples[None], (gen_samples.shape[0], 1, 1)), window_size=window_size)
            gt_intra_diversity.append(_gt_intra_diversity)
            print('gt_intra_diversity [{:.3f}]'.format(_gt_intra_diversity))
            _inter_diversity = calc_inter_diversity(eval_wrapper, gen_samples)
            inter_diversity.append(_inter_diversity)
            print('inter_diversity [{:.3f}]'.format(_inter_diversity))
            _sifid = calc_sifid(eval_wrapper, gen_samples, gt_samples, window_size=window_size)
            sifid.append(_sifid)
            print('SiFID [{:.3f}]'.format(_sifid))

        results[window_size] = {
            'intra_diversity': {'mean': np.mean(intra_diversity), 'std': np.std(intra_diversity)},
            'gt_intra_diversity': {'mean': np.mean(gt_intra_diversity), 'std': np.std(gt_intra_diversity)},
            'inter_diversity': {'mean': np.mean(inter_diversity), 'std': np.std(inter_diversity)},
            'sifid': {'mean': np.mean(sifid), 'std': np.std(sifid)},
        }

        print(f'===Summary [window_size={window_size}]===')
        print('intra_diversity [{:.3f}±{:.3f}]'.format(results[window_size]['intra_diversity']['mean'], results[window_size]['intra_diversity']['std']))
        print('gt_intra_diversity [{:.3f}±{:.3f}]'.format(results[window_size]['gt_intra_diversity']['mean'], results[window_size]['gt_intra_diversity']['std']))
        print('inter_diversity [{:.3f}±{:.3f}]'.format(results[window_size]['inter_diversity']['mean'], results[window_size]['inter_diversity']['std']))
        print('SiFID [{:.3f}±{:.3f}]'.format(results[window_size]['sifid']['mean'], results[window_size]['sifid']['std']))

    return results

if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_{}_{}'.format(name, niter))
    log_file += f'_{args.eval_mode}'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')

    print(f'Eval mode [{args.eval_mode}]')
    num_samples_limit = 100
    args.batch_size = num_samples_limit
    replication_times = 5

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = 'test'

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, None)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    model.eval()  # disable random masking

    logger.log(f"Loading evaluator")
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())

    evaluate(args, model, diffusion, eval_wrapper, num_samples_limit, replication_times)