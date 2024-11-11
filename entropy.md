random entropy stuff



When we are trying to maximize entropy, we are trying to maximize the absolute value of each term that is being summed, as entropy is a summation operation. Let’s use this form of the entropy equation as it’s the simplest:

$$H = -\sum_{i=1}^{N} p_i \log_2 p_i$$

So we are trying to maximize $$|p_i log_2 p_i|$$ for i = 1  to  N. If we let x = p_i and f(x) = log_2(x), we can calculate each term as x * f(x)   . Looking at a graph of   y = f(x)    which is a log graph as y = \log_2(x)   , the values of   x we are interested in is the interval   x in [0, 1]    since   x    is the probability, and a probability can only take on values in that interval. 

As   x    approaches 1 (ie as x is maximized),   f(x) approaches 0 (ie f(x) is minimized). When |x| is at its max value, 1, |y| is at its min value, 0. Similarly, when   |y|    is maximized, |x| is minimized. This is when x   approaches 0; there is no valid y value when   x = 0    as it is an asymptote ( since   log_2 0 = y    is undefined). (See the above images for visual representation of this inverse behaviour). Because of this behavior, the maximum absolute value for each term in the entropy equation, expressed here as x * f(x), should be somewhere in the middle of 0 and 1 i.e., any increase or decrease beyond the optimal point will result in a smaller entropy value.

Assuming all probabilities are equal,

$$ entropy = - \sum_{i=1}^{N} p_i \log_2 p_i \$$
$$ = - \left( N \cdot \frac{1}{N} \log_2 \frac{1}{N} \right) = -\log_2 \frac{1}{N}$$

Now, if we adjust the probabilities such that it is larger, the rest will be smaller. This will lead to a decrease in the absolute value of xf(x), which leads to an overall smaller entropy. Therefore, entropy is maximized when all symbols are equiprobable, with 1/N probability.