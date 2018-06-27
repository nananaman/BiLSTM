from pathlib import Path
import chainer


def out_generated_title(iter, model):
    @chainer.training.make_extension()
    def make_title(trainer):
        iter_length = iter.dataset.__len__()
        abstracts = []
        titles = []
        pred_titles = []
        i = 1
        for batch in iter:
            with chainer.using_config('train', False), chainer.function.no_backprop_mode():
                pred_title = model.predict(batch)
            abstract = batch[0][0]
            title = batch[0][1]
            abstracts.append(abstract)
            titles.append(title)
            pred_titles.append(pred_title)
            if i < iter_length:
                i += iter.batch_size
            else:
                break

        preview_dir = './result/preview'
        if not Path(preview_dir).exists():
            Path(preview_dir).mkdir(parents=True)
        dst = Path(preview_dir + '/epoch{0}.txt'.format(trainer.updater.epoch))

        with dst.open(mode='w', encoding='utf-8') as fout:
            for p, t, a in zip(pred_titles, titles, abstracts):
                fout.write(
                    '{0} : {1} \n {2}\n##############################\n'.format(p, t, a))
    return make_title
