// Perceptron Object
function Perceptron(no, learningRate = 0.00001) {

  // Set Initial Values
  this.learnc = learningRate;
  this.bias = 1;

  // Compute Random Weights
  this.weights = [];
  for (let i = 0; i <= no; i++) {
    this.weights[i] = Math.random() * 2 - 1;
  }

  // Activate Function
  this.activate = function(inputs) {
    let sum = 0;
    for (let i = 0; i < inputs.length; i++) {
      sum += inputs[i] * this.weights[i];
    }
    return sum > 0 ? 1 : 0;
  }

  // Train Function
  this.train = function(inputs, desired) {
    const inputsWithBias = [...inputs, this.bias]; // Clone + add bias
    const guess = this.activate(inputsWithBias);
    const error = desired - guess;
    if (error !== 0) {
      for (let i = 0; i < inputsWithBias.length; i++) {
        this.weights[i] += this.learnc * error * inputsWithBias[i];
      }
    }
  }

  // Predict function (optional)
  this.predict = function(inputs) {
    return this.activate([...inputs, this.bias]);
  }
}
