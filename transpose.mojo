from tensor import Tensor, TensorShape
alias type = DType.float32

# Convert coordinate list to index value
fn Index(dimensions: List[Int], nums: List[Int]) -> Int:
  var idx: Int = 0
  var mul: Int = 1

  for i in range(len(dimensions)):
    idx += mul * nums[i]
    mul *= dimensions[i]

  return idx

# Convert index value to coordinate list
fn Coordinate(dimensions: List[Int], idx: Int) -> List[Int]:
  var nums: List[Int] = List[Int]()
  var div: Int = 1
  var index: Int = idx

  for i in dimensions:
    div *= i[]
  
  for i in dimensions:
    div /= i[]
    nums.append(index // div)
    index %= div

  return nums

# Transpose 2 dimensions of a tensor
fn tensor_transpose(container: Tensor[type], dim1: Int, dim2: Int) -> Tensor[type]:
  var original_shape: List[Int] = List[Int]()
  var new_shape: List[Int]
  var output: Tensor[type]
  var coordinates: List[Int]

  # Creating tensor with switched dimensions
  for i in range(container.shape().rank()):
    original_shape.append(container.dim(i))
  
  new_shape = original_shape
  new_shape[dim1], new_shape[dim2] = new_shape[dim2], new_shape[dim1]

  output = Tensor[type](TensorShape(new_shape))

  # Transposing by iterating through each tensor element
  for i in range(container.num_elements()):
    coordinates = Coordinate(original_shape, i)
    coordinates[dim1], coordinates[dim2] = coordinates[dim2], coordinates[dim1]
    output[Index(new_shape, coordinates)] = container[i]
   
  return output
