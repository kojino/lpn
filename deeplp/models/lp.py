import numpy as np
import scipy as sp
from deeplp.models.utils import rbf_kernel
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import softmax

class LP:
    """
    Label propagation for predicting labels for unlabeled nodes.
    Closed form and iterated solutions.
    See mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf for details.
    """

    def closed(self, labels,
                     weights,
                     labeled_indices,
                     unlabeled_indices):
        """Closed solution of label propagation
        Input:
            labels: one-hot encoded labels
            weights: weight adjacency matrix
            labeled_indices: indices of labeled nodes
            unlabeled_indices: indices of unlabeled nodes
        Returns:
            label predictions for the unlabeled nodes
        """
        # normalize T
        Tnorm = self._tnorm(weights)
        # sort Tnorm by unlabeled/labeld
        Tuu_norm = Tnorm[np.ix_(unlabeled_indices,unlabeled_indices)]
        Tul_norm = Tnorm[np.ix_(unlabeled_indices,labeled_indices)]
        # closed form prediction for unlabeled nodes
        lapliacian = (np.identity(len(Tuu_norm))-Tuu_norm)
        propagated = Tul_norm @ labels[labeled_indices]
        label_predictions = np.linalg.solve(lapliacian, propagated)
        return label_predictions

    def closed_sp(self, labels,
                        weights,
                        labeled_indices,
                        unlabeled_indices):
        """Closed solution of label propagation
        Input:
            labels: one-hot encoded labels
            weights: weight adjacency matrix
            labeled_indices: indices of labeled nodes
            unlabeled_indices: indices of unlabeled nodes
        Returns:
            label predictions for the unlabeled nodes
        """
        # normalize T
        Tnorm = normalize(weights, norm='l1', axis=1)
        # sort Tnorm by unlabeled/labeld
        Tuu_norm = Tnorm[np.ix_(unlabeled_indices,unlabeled_indices)]
        Tul_norm = Tnorm[np.ix_(unlabeled_indices,labeled_indices)]

        # closed form prediction for unlabeled nodes
        lapliacian = sp.sparse.identity(Tuu_norm.shape[0]) - Tuu_norm
        propagated = Tul_norm @ labels[labeled_indices]
        label_predictions = sp.sparse.linalg.spsolve(lapliacian, propagated)
        return label_predictions

    def iter(self, labels, # input labels
                   weights,
                   is_labeled,
                   num_iter,
                   unlabeled_indices,
                   clamp=0,
                   laplacian=0):
        """Iterated solution of label propagation.
        Input:
            weights: weight adjacency matrix
            is_labeled: boolean indicating labeled nodes
            num_iter: number of iterations of propagation
            unlabeled_indices: indices of unlabeled nodes
        Returns:
            label predictions for the unlabeled nodes
        """
        # normalize T
        Tnorm = self._tnorm(weights)

        h = labels.copy()

        D = np.zeros(weights.shape)
        np.fill_diagonal(D, np.sum(np.abs(weights),axis=0,keepdims=True))
        Dinv = np.linalg.inv(D)

        if laplacian:
            Dinv_sqrt = np.sqrt(Dinv)
            A = Dinv_sqrt @ weights @ Dinv_sqrt

        for i in range(num_iter):
            if not laplacian:
                # propagate labels
                h = Tnorm @ h
                # don't update labeled nodes
                h = h * (1-is_labeled) + labels * is_labeled
            else:
                h = (1-laplacian) * A @ h + laplacian * labels

            if clamp:
                h = softmax(clamp * h)

        # only return label predictions
        return h[unlabeled_indices]


    def iter_sp(self, labels, # input labels
                   weights,
                   is_labeled,
                   num_iter,
                   unlabeled_indices,
                   clamp=0,
                   laplacian=0):
        """Iterated solution of label propagation.
        Input:
            weights: weight adjacency matrix
            is_labeled: boolean indicating labeled nodes
            num_iter: number of iterations of propagation
            unlabeled_indices: indices of unlabeled nodes
        Returns:
            label predictions for the unlabeled nodes
        """
        # normalize T
        Tnorm = normalize(weights, norm='l1', axis=1)
        vals=Tnorm.tocoo().data
        h = labels.copy()

        if laplacian:
            diagonals = np.sum(np.abs(weights),axis=1).T
            offset = [0]
            D = sp.sparse.diags([np.array(np.abs(weights).sum(axis=1).T)[0]], [0])
            Dinv = sp.sparse.linalg.inv(D.tocsc())

            Dinv_sqrt = np.sqrt(Dinv)
            A = Dinv_sqrt @ weights @ Dinv_sqrt

        for i in range(num_iter):
            if not laplacian:
                # propagate labels
                h = Tnorm @ h
                # don't update labeled nodes
                h = h * (1-is_labeled) + labels * is_labeled
            else:
                h = (1-laplacian) * A @ h + laplacian * labels

            if clamp:
                h = softmax(clamp * h)

        # only return label predictions
        return h[unlabeled_indices]


    def _tnorm(self,weights):
        """Column normalize weights"""
        # row normalize T
        Tnorm = weights / np.sum(weights, axis=1, keepdims=True)
        return Tnorm
