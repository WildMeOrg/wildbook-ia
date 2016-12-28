def load_multiclass_scores(self):
    # convert simple scores to multiclass scores
    import vtool as vt
    self.multiclass_scores = {}
    for key in self.samples.simple_scores.keys():
        scores = self.samples.simple_scores[key].values
        # Hack scores into the range 0 to 1
        normer = vt.ScoreNormalizer(adjust=8, monotonize=True)
        normer.fit(scores, y=self.samples.is_same())
        normed_scores = normer.normalize_scores(scores)
        # Create a dimension for each class
        # but only populate two of the dimensions
        class_idxs = ut.take(self.samples.text_to_class, ['nomatch', 'match'])
        pred = np.zeros((len(scores), len(self.samples.class_names)))
        pred[:, class_idxs[0]] = 1 - normed_scores
        pred[:, class_idxs[1]] = normed_scores
        self.multiclass_scores[key] = pred
