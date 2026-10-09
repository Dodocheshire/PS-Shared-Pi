"""Keep an observed feature view and its frozen conditioning concepts aligned."""


def select_training_views(features, contexts, ids, views):
    if features.ndim!=3 or contexts.ndim!=3 or features.shape[:2]!=contexts.shape[:2]:
        raise ValueError('Feature/context view pools differ')
    if ids.ndim!=1 or views.shape!=ids.shape:
        raise ValueError('Expected one view choice per original image')
    return features[views,ids],contexts[views,ids]
