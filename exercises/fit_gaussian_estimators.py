import numpy.random

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariate = UnivariateGaussian()
    random_1000 = numpy.random.normal(10, 1, 1000)
    univariate.fit(random_1000)
    print("({exp}, {var})".format(exp=univariate.mu_, var=univariate.var_))

    # Question 2 - Empirically showing sample mean is consistent
    # raise NotImplementedError()

    list_rand = [univariate.fit(random_1000[0:i]).mu_ for i in range(10, 1010, 10)]
    np_random = np.abs(np.array(list_rand) - 10)
    data = go.Scatter(x=list(range(10, 1010, 10)), y=np_random, mode='markers')
    figure = go.Figure(data)
    figure.update_layout(xaxis_title="range 10 to 1000", yaxis_title="distance between estimated and true value of "
                                                                     "expectation")
    figure.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()

    random_pdf = univariate.pdf(random_1000)
    data = go.Scatter(x=random_1000, y=random_pdf, mode='markers')
    figure = go.Figure(data)
    # figure.update_layout(xaxis_title="random normal", yaxis_title="univariate gaussian PDF")
    figure.show()

def create_cartesian_product(vec1,vec2):
    len3 = len(vec1) * len(vec2)
    return np.transpose(np.array([np.repeat(vec1, len(vec2)), np.zeros(len3), np.tile(vec2, len(vec1)), np.zeros(len3)]))

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    # raise NotImplementedError()
    multivariate = MultivariateGaussian()
    multi_mu = np.array([0, 0, 4, 0])
    multi_sigma = np.array([[1, 0.2, 0, 0.5],
                            [0.2, 2, 0, 0],
                            [0, 0, 1, 0],
                            [0.5, 0, 0, 1]])
    x = numpy.random.multivariate_normal(multi_mu, multi_sigma, 1000)
    multivariate.fit(x)
    print(multivariate.mu_)
    print(multivariate.cov_)
    # multivariate.log_likelihood(multivariate.mu_, multivariate.cov_, x)

    # Question 5 - Likelihood evaluation
    # raise NotImplementedError()
    f1 = np.linspace(-10, 10, 200)
    vectors = create_cartesian_product(f1, f1)
    log_vectors = np.array([multivariate.log_likelihood(vectors[i, :], multi_sigma, x) for i in range(len(f1)**2)])
    fig = go.Figure(data = go.Heatmap(x=np.repeat(f1, len(f1)), y=np.tile(f1, len(f1)), z=log_vectors))
    fig.show()

    # Question 6 - Maximum likelihood
    # raise NotImplementedError()
    print(np.around(vectors[np.argmax(log_vectors)], 3))


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()

#     random_pdf = univariate.pdf(random_1000)
#     data = go.Scatter(x=random_1000, y=random_pdf, mode='markers')
#     figure = go.Figure(data)
#     figure.update_layout(xaxis_title="random normal", yaxis_title="univariate gaussian PDF")
#     figure.show()
