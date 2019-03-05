// Keras-Sharp: C# port of the Keras library
// https://github.com/cesarsouza/keras-sharp
//
// Based under the Keras library for Python. See LICENSE text for more details.
//
//    The MIT License(MIT)
//    
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//    
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//    
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.
//

namespace KerasSharp.Backends
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using KerasSharp.Engine.Topology;
    using KerasSharp.Losses;
    using KerasSharp.Models;
    using Tensorflow;
    using static KerasSharp.Python;
    using Accord.Math;

    public class TensorFlowBackend : BackendBase, IBackend
    {
        internal Graph tf;


        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L25

        // This is the default internal TF session used by Keras.
        // It can be set manually via `set_session(sess)`.
        internal Session _SESSION;

        // This dictionary holds a mapping {graph: learning_phase}.
        // A learning phase is a bool tensor used to run Keras models in
        // either train mode (learning_phase == 1) or test mode (learning_phase == 0).
        private Dictionary<Graph, object> _GRAPH_LEARNING_PHASES = new Dictionary<Graph, object>();

        // This dictionary holds a mapping {graph: UID_DICT}.
        // each UID_DICT is a dictionary mapping name prefixes to a current index,
        // used for generatic graph-specific string UIDs
        // for various names (e.g. layer names).
        private Dictionary<Graph, Dictionary<string, int>> _GRAPH_UID_DICTS = new Dictionary<Graph, Dictionary<string, int>>();

        // This boolean flag can be set to True to leave variable initialization
        // up to the user.
        // Change its value via `manual_variable_initialization(value)`.
        bool _MANUAL_VAR_INIT = false;


        public TensorFlowBackend()
        {
            this.tf = new Graph();
            this._SESSION = new Session(tf);
        }

        /// <summary>
        ///   Get the uid for the default graph.
        /// </summary>
        /// 
        /// <param name="prefix">An optional prefix of the graph.</param>
        /// 
        /// <returns>A unique identifier for the graph.</returns>
        /// 
        public int get_uid(string prefix)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L58
            var graph = tf;
            if (!_GRAPH_UID_DICTS.ContainsKey(graph))
                _GRAPH_UID_DICTS[graph] = new Dictionary<string, int>();
            if (!_GRAPH_UID_DICTS[graph].ContainsKey(prefix))
                _GRAPH_UID_DICTS[graph][prefix] = 0;
            _GRAPH_UID_DICTS[graph][prefix] += 1;
            return _GRAPH_UID_DICTS[graph][prefix];
        }

        /// <summary>
        ///   Reset graph identifiers.
        /// </summary>
        /// 
        public void reset_uids()
        {
            _GRAPH_UID_DICTS = new Dictionary<Graph, Dictionary<string, int>>();
        }

        /// <summary>
        ///   Destroys the current TF graph and creates a new one.
        ///   Useful to avoid clutter from old models / layers.
        /// </summary>
        /// 
        public void clear_session()
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L71
            // tf.reset_default_graph();
            tf = new Graph();
            _SESSION = new Session(tf);
            //
            reset_uids();
            TF_Output phase = Tensorflow.tf.placeholder(dtype: TF_DataType.TF_BOOL, operName: "keras_learning_phase");
            _GRAPH_LEARNING_PHASES = new Dictionary<Graph, object>();
            _GRAPH_LEARNING_PHASES[tf] = phase;
        }





        /// <summary>
        ///   Reshapes a tensor to the specified shape.
        /// </summary>
        /// 
        /// <param name="x">The Tensor or variable.</param>
        /// <param name="shape">The target shape.</param>
        /// 
        /// <returns>Tensor.</returns>
        /// 
        public KerasSharp.Engine.Topology.Tensor reshape(KerasSharp.Engine.Topology.Tensor x, int[] shape)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L1724
            return Out(gen_array_ops.reshape(In(x).tensor, In(shape).tensor));
        }



        public KerasSharp.Engine.Topology.Tensor abs(KerasSharp.Engine.Topology.Tensor input)
        {
            return Out(tf.Abs(In(input)));
        }






        public List<Array> batch_get_value(List<KerasSharp.Engine.Topology.Tensor> weights)
        {
            throw new NotImplementedException();
        }

        public List<Array> batch_get_value(List<List<KerasSharp.Engine.Topology.Tensor>> weights)
        {
            throw new NotImplementedException();
        }

        public void batch_set_value(List<Tuple<KerasSharp.Engine.Topology.Tensor, Array>> weight_value_tuples)
        {
            throw new NotImplementedException();
        }

        public void batch_set_value(List<(KerasSharp.Engine.Topology.Tensor, Array)> tuples)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Binary crossentropy between an output tensor and a target tensor.
        /// </summary>
        /// 
        /// <param name="output">A tensor.</param>
        /// <param name="target">A tensor of the same shape as `output`.</param>
        /// <param name="from_logits">Whether `output` is expected to be a logits tensor. By default, we consider that `output` encodes a probability distribution.</param>
        /// 
        /// <returns>Output tensor.</returns>
        /// 
        public KerasSharp.Engine.Topology.Tensor binary_crossentropy(KerasSharp.Engine.Topology.Tensor output, KerasSharp.Engine.Topology.Tensor target, bool from_logits = false)
        {
            TF_Output _output = In(output);
            TF_Output _target = In(target);
            TF_DataType dtype = _output.OutputType;

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2792

            // Note: tf.nn.sigmoid_cross_entropy_with_logits
            // expects logits, Keras expects probabilities.
            if (!from_logits)
            {
                // transform back to logits
                TF_Output _epsilon = _constant(epsilon(), dtype: dtype);
                _output = tf.ClipByValue(_output, _epsilon, gen_math_ops.sub(_constant(1, dtype: dtype), _epsilon));
                _output = tf.Log(gen_math_ops.real_div(_output, gen_math_ops.sub(_constant(1, dtype: dtype), _output)));
            }

            return Out(tf.SigmoidCrossEntropyWithLogits(labels: _target, logits: _output));
        }

        public KerasSharp.Engine.Topology.Tensor cast(KerasSharp.Engine.Topology.Tensor x, DataType dataType)
        {
            return Out(Tensorflow.tf.cast(In(x).tensor, In(dataType)));
        }

        /// <summary>
        ///   Categorical crossentropy between an output tensor and a target tensor.
        /// </summary>
        /// 
        /// <param name="target">A tensor of the same shape as `output`.</param>
        /// <param name="output">A tensor resulting from a softmax (unless `from_logits` is True, in which case `output` is expected to be the logits).</param>
        /// <param name="from_logits">Boolean, whether `output` is the result of a softmax, or is a tensor of logits.</param>
        /// 
        /// <returns>Output tensor.</returns>
        /// 
        public KerasSharp.Engine.Topology.Tensor categorical_crossentropy(KerasSharp.Engine.Topology.Tensor target, KerasSharp.Engine.Topology.Tensor output, bool from_logits = false)
        {
            var _target = In(target);
            var _output = In(output);

            // Note: tf.nn.softmax_cross_entropy_with_logits
            // expects logits, Keras expects probabilities.
            if (!from_logits)
            {
                // scale preds so that the class probas of each sample sum to 1
                int?[] shape = output.shape;
                var last = Tensorflow.tf.constant(new Tensorflow.Tensor(shape.Length - 1));
                TF_Output o = Tensorflow.tf.divide(_output, Tensorflow.tf.reduce_sum(_output, axis: last));
                // manual computation of crossentropy
                TF_Output _epsilon = _constant(epsilon(), dtype: _output.dtype);
                o = tf.ClipByValue(o, _epsilon, Tensorflow.tf.sub(Tensorflow.tf.constant(1f), Tensorflow.tf.constant(_epsilon)));
                o = tf.Neg(Tensorflow.tf.reduce_sum(Tensorflow.tf.multiply(_target, tf.Log(_output)), axis: last));
                return Out(o);
            }

            return Out(tf.SoftmaxCrossEntropyWithLogits(_target, _output).loss);
        }

        public KerasSharp.Engine.Topology.Tensor clip(KerasSharp.Engine.Topology.Tensor norms, int v, int maxValue)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor clip(KerasSharp.Engine.Topology.Tensor norms, double min_value, double max_value)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor clip_norm(KerasSharp.Engine.Topology.Tensor g, double clipnorm, KerasSharp.Engine.Topology.Tensor norm)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor constant<T>(T value, int[] shape = null, DataType? dtype = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            int[] _shape;
            if (shape == null)
            {
                Array arr = value as Array;
                if (arr != null)
                    _shape = arr.GetLength();
                else _shape = new int[0];
                shape = _shape;
            }
            else
            {
                _shape = shape;
            }

            TF_Output o;
            if (shape != null && shape.Length != 0 && !(value is Array))
            {
                o = _constant(Matrix.Create(value.GetType(), _shape, value), In(dtype.Value), name);
            }
            else
            {
                o = _constant(value, In(dtype.Value), name);
            }

            if (!_int_shape(o).IsEqual(shape))
                throw new Exception();

            return Out(o);
        }

        private TF_Output _constant<T>(T value, TF_DataType? dtype = null, string operName = null)
        {
            Tensorflow.Tensor t = new Tensorflow.Tensor((dynamic)value);

            TF_Output o = Tensorflow.tf.constant(t, name: operName)._as_tf_output();

            if (dtype == null || o.OutputType == dtype.Value)
                return o;

            return Tensorflow.tf.cast(t, dtype.Value)._as_tf_output();
        }




        public KerasSharp.Engine.Topology.Tensor dropout(object p, double retain_prob, object noise_shape, object seed)
        {
            throw new NotImplementedException();
        }

        public DataType? dtype(KerasSharp.Engine.Topology.Tensor tensor)
        {
            return Out(In(tensor).dtype);
        }

        public KerasSharp.Engine.Topology.Tensor elu(KerasSharp.Engine.Topology.Tensor x)
        {
            return Out(tf.Elu(In(x)));
        }

        public KerasSharp.Engine.Topology.Tensor elu(KerasSharp.Engine.Topology.Tensor x, double alpha)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor exp(KerasSharp.Engine.Topology.Tensor x)
        {
            return Out(tf.Exp(In(x)));
        }

        public KerasSharp.Models.Function function(List<KerasSharp.Engine.Topology.Tensor> inputs, List<KerasSharp.Engine.Topology.Tensor> outputs, List<List<KerasSharp.Engine.Topology.Tensor>> updates, string name)
        {
            return new TFFunction(this, inputs: inputs, outputs: outputs, updates: updates, name: name);
        }



        /// <summary>
        ///   Returns the shape of a variable.
        /// </summary>
        /// 
        public int?[] get_variable_shape(KerasSharp.Engine.Topology.Tensor x)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2192
            return int_shape(x);
        }

        public List<KerasSharp.Engine.Topology.Tensor> gradients(KerasSharp.Engine.Topology.Tensor loss, List<KerasSharp.Engine.Topology.Tensor> param)
        {
            var y = new TF_Output[] { In(loss).output };
            var x = param.Select(t => In(t).output).ToArray();

            TF_Output[] grads = tf.AddGradients(y, x);

            List<KerasSharp.Engine.Topology.Tensor> r = new List<KerasSharp.Engine.Topology.Tensor>();
            for (int i = 0; i < grads.Length; i++)
                r.Add(Out(grads[i], name: "grad/" + x[i].Operation.Name));

            // TODO: TensorFlow.NET: convert to Tensor[]???
            //return Tensorflow.tf.gradients(In(loss).tensor, param, colocate_gradients_with_ops: true);

            return r;
        }

        public KerasSharp.Engine.Topology.Tensor greater_equal(KerasSharp.Engine.Topology.Tensor w, double v)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor not_equal(KerasSharp.Engine.Topology.Tensor x, KerasSharp.Engine.Topology.Tensor y)
        {
            return Out(tf.NotEqual(In(x), In(y)));
        }

        public KerasSharp.Engine.Topology.Tensor not_equal<T>(KerasSharp.Engine.Topology.Tensor x, T y) where T : struct
        {
            using (this.name_scope("not_equal"))
            {
                TensorFlowTensor _x = In(x);
                var _y = Tensorflow.tf.cast(Tensorflow.tf.constant((dynamic)y), _x.dtype);
                return Out(tf.NotEqual(_x, _y));
            }
        }

        public KerasSharp.Engine.Topology.Tensor hard_sigmoid(KerasSharp.Engine.Topology.Tensor x)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor identity(KerasSharp.Engine.Topology.Tensor x, string name = null)
        {
            return Out(gen_array_ops.identity(In(x).tensor, name: name));
        }

        /// <summary>
        ///   Returns the shape tensor or variable as a tuple of int or None entries.
        /// </summary>
        /// 
        /// <param name="x">Tensor or variable.</param>
        /// 
        /// <returns>A tuple of integers(or None entries).</returns>
        /// 
        public int?[] int_shape(KerasSharp.Engine.Topology.Tensor x)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L468

            if (x._keras_shape != null)
                return x._keras_shape;

            return _int_shape(In(x).output);
        }

        private int?[] _int_shape(TF_Output _x)
        {
            try
            {
                long[] shape = tf.GetTensorShape(_x).ToArray();
                return shape.Select(i => i == -1 ? null : (int?)i).ToArray();
            }
            catch
            {
                return null;
            }
        }

        public int?[] int_shape(Tensorflow.Tensor input_tensor)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Selects `x` in train phase, and `alt` otherwise.
        /// </summary>
        /// 
        /// <param name="x">What to return in train phase.</param>
        /// <param name="alt">What to return otherwise.</param>
        /// <param name="training">Optional scalar tensor specifying the learning phase.</param>
        /// 
        /// <returns>Either 'x' or 'alt' based on the 'training' flag. The 'training' flag defaults to 'K.learning_phase()'.</returns>
        /// 
        public KerasSharp.Engine.Topology.Tensor in_train_phase(Func<KerasSharp.Engine.Topology.Tensor> x, Func<KerasSharp.Engine.Topology.Tensor> alt, bool? training)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2583

            bool uses_learning_phase;

            if (training == null)
            {
                var t = learning_phase();
                if (t is bool)
                    training = (bool)t;
                uses_learning_phase = true;
            }
            else
            {
                uses_learning_phase = false;
            }

            if (training == true)
            {
                return x();
            }
            else if (training == false)
            {
                return alt();
            }
            else
            {
                //else: assume learning phase is a placeholder tensor.

                KerasSharp.Engine.Topology.Tensor xx = @switch((KerasSharp.Engine.Topology.Tensor)learning_phase(), x, alt);

                if (uses_learning_phase)
                    xx._uses_learning_phase = true;
                return xx;
            }
        }

        /// <summary>
        ///   Selects `x` in test phase, and `alt` otherwise. Note that `alt` should have the* same shape* as `x`.
        /// </summary>
        public KerasSharp.Engine.Topology.Tensor in_test_phase(Func<KerasSharp.Engine.Topology.Tensor> x, Func<KerasSharp.Engine.Topology.Tensor> alt, bool? training = null)
        {
            return in_train_phase(alt, x, training: training);
        }

        /// <summary>
        ///   Switches between two operations depending on a scalar value. Note that both `then_expression` and `else_expression`
        ///   should be symbolic tensors of the *same shape
        /// </summary>
        /// 
        /// <param name="condition">The condition: scalar tensor(`int` or `bool`).</param>
        /// <param name="then_expression">Either a tensor, or a callable that returns a tensor.</param>
        /// <param name="else_expression">Either a tensor, or a callable that returns a tensor.</param>
        /// 
        /// <returns>The selected tensor.</returns>
        /// 
        public KerasSharp.Engine.Topology.Tensor @switch(KerasSharp.Engine.Topology.Tensor condition, Func<KerasSharp.Engine.Topology.Tensor> then_expression, Func<KerasSharp.Engine.Topology.Tensor> else_expression)
        {
            var _condition = In(condition);

            if (_condition.dtype != TF_DataType.TF_BOOL)
                condition = Out(Tensorflow.tf.cast(_condition.tensor, TF_DataType.TF_BOOL));

            TF_Output x = tf.Cond(In(condition),
                        () => In(then_expression()),
                        () => In(else_expression()));
            return Out(x);
        }

        public bool is_sparse(KerasSharp.Engine.Topology.Tensor tensor)
        {
            return false;
        }

        public KerasSharp.Engine.Topology.Tensor l2_normalize(KerasSharp.Engine.Topology.Tensor expected, int axis)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Returns the learning phase flag.
        /// </summary>
        /// 
        /// <remarks>
        ///   The learning phase flag is a bool tensor(0 = test, 1 = train)
        ///   to be passed as input to any Keras function
        ///   that uses a different behavior at train time and test time.
        /// </remarks>
        /// 
        /// <returns> Learning phase (scalar integer tensor or Python integer).</returns>
        /// 
        public object learning_phase()
        {
            Graph graph = tf;
            if (!_GRAPH_LEARNING_PHASES.ContainsKey(graph))
            {
                TF_Output phase = Tensorflow.tf.placeholder(dtype: TF_DataType.TF_BOOL, operName: "keras_learning_phase");
                _GRAPH_LEARNING_PHASES[graph] = phase;
            }

            return _GRAPH_LEARNING_PHASES[graph];
        }

        /// <summary>
        ///   Sets the learning phase to a fixed value.
        /// </summary>
        public void set_learning_phase(bool value)
        {
            _GRAPH_LEARNING_PHASES[tf] = value;
        }

        public KerasSharp.Engine.Topology.Tensor max(KerasSharp.Engine.Topology.Tensor x, int v, object p)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor max(KerasSharp.Engine.Topology.Tensor x, int axis, bool keepdims)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor max(KerasSharp.Engine.Topology.Tensor tensor, int axis)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor maximum(double v, KerasSharp.Engine.Topology.Tensor tensor)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Turn a nD tensor into a 2D tensor with same 0th dimension. In other words, it flattens each data samples of a batch.
        /// </summary>
        /// 
        public KerasSharp.Engine.Topology.Tensor batch_flatten(KerasSharp.Engine.Topology.Tensor x)
        {
            var _x = In(x);
            TF_Output shape = tf.Shape(_x);
            
            TF_Output dim = tf.Prod(tf.Slice(shape, Tensorflow.tf.constant(1), array_ops.rank(_x.tensor)), reduction_indices: tf.ReduceDims(shape, null));
            // TODO: TensorFlow.NET
            //gen_array_ops.reshape(In(x).tensor, TODO);
            return Out(tf.Reshape(In(x), tf.Stack(new TF_Output[] { Tensorflow.tf.constant(-1)._as_tf_output(), dim } )));
        }


        public int[] _normalize_axis(int[] axis, int? ndim)
        {
            axis = (int[])axis.Clone();
            for (int i = 0; i < axis.Length; i++)
            {
                if (axis[i] < 0)
                    axis[i] = axis[i] % ndim.Value;
            }

            return axis;
        }

        /// <summary>
        ///   Mean of a tensor, alongside the specified axis.
        /// </summary>
        /// 
        /// <param name="x">A tensor or variable.</param>
        /// <param name="axis">A list of integer. Axes to compute the mean.</param>
        /// <param name="keepdims>A boolean, whether to keep the dimensions or not. If <paramref name="keepdims"/> is <c>false</c>, 
        ///   the rank of the tensor is reduced by 1 for each entry in <paramref name="axis"/>. If <paramref name="keepdims"/> is 
        ///   <c>true</c>, the reduced dimensions are retained with length 1.
        ///   
        /// <returns>A tensor with the mean of elements of <c>x</c>.</returns>
        /// 
        public KerasSharp.Engine.Topology.Tensor mean(KerasSharp.Engine.Topology.Tensor x, int[] axis, bool keepdims = false, string name = null)
        {
            return Out(math_ops.reduce_mean(In(x).tensor, _normalize_axis(axis, ndim(x)), keepdims, name: name));
        }

        /// <summary>
        ///   Mean of a tensor, alongside the specified axis.
        /// </summary>
        /// 
        /// <param name="x">A tensor or variable.</param>
        /// <param name="axis">The axis where to compute the mean.</param>
        /// <param name="keepdims>A boolean, whether to keep the dimensions or not. If <paramref name="keepdims"/> is <c>false</c>, 
        ///   the rank of the tensor is reduced by 1 for each entry in <paramref name="axis"/>. If <paramref name="keepdims"/> is 
        ///   <c>true</c>, the reduced dimensions are retained with length 1.
        ///   
        /// <returns>A tensor with the mean of elements of <c>x</c>.</returns>
        /// 
        public KerasSharp.Engine.Topology.Tensor mean(KerasSharp.Engine.Topology.Tensor x, int axis = -1, bool keepdims = false, string name = null)
        {
            return Out(math_ops.reduce_mean(In(x).tensor, axis: Tensorflow.tf.constant(axis), keepdims: keepdims, name: name));
        }


        public KerasSharp.Engine.Topology.Tensor minus(KerasSharp.Engine.Topology.Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor dot(KerasSharp.Engine.Topology.Tensor a, KerasSharp.Engine.Topology.Tensor b, string name = null)
        {
            return Out(math_ops.matmul(In(a).tensor, In(b).tensor, name: name));
        }


        public KerasSharp.Engine.Topology.Tensor mul<T>(T a, KerasSharp.Engine.Topology.Tensor b, string name = null)
        {
            return mul(constant(a, dtype: dtype(b)), b, name: name);
        }

        public KerasSharp.Engine.Topology.Tensor mul(KerasSharp.Engine.Topology.Tensor a, KerasSharp.Engine.Topology.Tensor b, string name = null)
        {
            return Out(gen_math_ops.mul(In(a).tensor, In(b).tensor, name: name));
        }

        public KerasSharp.Engine.Topology.Tensor mul<T>(KerasSharp.Engine.Topology.Tensor a, T b, string name = null)
        {
            return mul(a, constant(b, dtype: dtype(a), name: name));
        }

        public KerasSharp.Engine.Topology.Tensor mul(List<KerasSharp.Engine.Topology.Tensor> batch_outs, int length)
        {
            throw new NotImplementedException();
        }





        public KerasSharp.Engine.Topology.Tensor div<T>(T a, KerasSharp.Engine.Topology.Tensor b)
        {
            return div(constant(a, dtype: dtype(b)), b);
        }

        public KerasSharp.Engine.Topology.Tensor div(KerasSharp.Engine.Topology.Tensor a, KerasSharp.Engine.Topology.Tensor b)
        {
            return Out(gen_math_ops.real_div(In(a).tensor, In(b).tensor));
        }

        public KerasSharp.Engine.Topology.Tensor div<T>(KerasSharp.Engine.Topology.Tensor a, T b)
        {
            return div(a, constant(b, dtype: dtype(a)));
        }

        public KerasSharp.Engine.Topology.Tensor div(List<KerasSharp.Engine.Topology.Tensor> batch_outs, int length)
        {
            throw new NotImplementedException();
        }



        public KerasSharp.Engine.Topology.Tensor add(KerasSharp.Engine.Topology.Tensor a, KerasSharp.Engine.Topology.Tensor b)
        {
            return Out(math_ops.add(In(a).tensor, In(b).tensor));
        }

        public KerasSharp.Engine.Topology.Tensor bias_add(KerasSharp.Engine.Topology.Tensor a, KerasSharp.Engine.Topology.Tensor b, DataFormatType? data_format = null, string name = null)
        {
            return Out(tf.BiasAdd(In(a), In(b), data_format: In(data_format), operName: name));
        }

        private string In(DataFormatType? data_format)
        {
            if (data_format == null)
                return null;

            switch (data_format.Value)
            {
                case DataFormatType.ChannelsFirst:
                    return "channels_first";
                case DataFormatType.ChannelsLast:
                    return "channels_last";
                default:
                    throw new Exception();
            }
        }

        public KerasSharp.Engine.Topology.Tensor add<T>(T a, KerasSharp.Engine.Topology.Tensor b)
        {
            return add(constant(a), b);
        }

        public KerasSharp.Engine.Topology.Tensor add<T>(KerasSharp.Engine.Topology.Tensor a, T b)
        {
            return add(a, constant(b));
        }



        public KerasSharp.Engine.Topology.Tensor subtract(KerasSharp.Engine.Topology.Tensor a, KerasSharp.Engine.Topology.Tensor b, string name = null)
        {
            return Out(gen_math_ops.sub(In(a).tensor, In(b).tensor, name: name));
        }

        public KerasSharp.Engine.Topology.Tensor subtract<T>(T a, KerasSharp.Engine.Topology.Tensor b, string name = null)
        {
            return subtract(constant(a), b, name: name);
        }

        public KerasSharp.Engine.Topology.Tensor subtract<T>(KerasSharp.Engine.Topology.Tensor a, T b, string name = null)
        {
            return subtract(a, constant(b), name: name);
        }



        public NameScope name_scope(string name)
        {
            return new TensorFlowNameScope(tf.WithScope(name), name);
        }

        public NameScope name_scope(string operName, string userName)
        {
            string name = MakeName(operName, userName);
            return new TensorFlowNameScope(tf.WithScope(name), name);
        }



        /// <summary>
        /// Returns the number of axes in a tensor, as an integer.
        /// </summary>
        /// <param name="x">Tensor or variable.</param>
        /// <example>
        /// <codesrc="TensorFlowBackendTest.cs" region="doc_ndim">
        /// </example>
        /// 
        public int? ndim(KerasSharp.Engine.Topology.Tensor x)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L519

            int?[] dims = x.shape;

            if (dims != null)
                return dims.Length;

            return tf.GetTensorNumDims(In(x).output);
        }

        public KerasSharp.Engine.Topology.Tensor placeholder(int?[] shape = null, int? ndim = null, DataType? dtype = null, bool sparse = false, string name = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L397

            if (sparse)
                throw new NotImplementedException();

            if (dtype == null)
                dtype = floatx();

            if (shape == null)
            {
                if (ndim != null)
                    shape = new int?[ndim.Value];
            }

            var tfshape = this.In(shape);

            KerasSharp.Engine.Topology.Tensor x = Out(Tensorflow.tf.placeholder(In(dtype.Value), tfshape, operName: name));
            x._keras_shape = shape;
            x._uses_learning_phase = false;
            return x;
        }

        /// <summary>
        ///   Returns a tensor with uniform distribution of values.
        /// </summary>
        /// <param name="shape">A tuple of integers, the shape of tensor to create.</param>
        /// <param name="minval">A float, lower boundary of the uniform distribution to draw samples.</param>
        /// <param name="maxval">A float, upper boundary of the uniform distribution to draw samples.</param>
        /// <param name="dtype">The dtype of returned tensor.</param>
        /// <param name="seed">The random seed.</param>
        /// 
        /// <returns>A tensor.</returns>
        /// 
        public KerasSharp.Engine.Topology.Tensor random_uniform(int[] shape, double minval = 0.0, double maxval = 1.0, DataType? dtype = null, int? seed = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            var _dtype = In(dtype.Value);

            if (seed == null)
                seed = Accord.Math.Random.Generator.Random.Next(1_000_000);

            using (name_scope("random_uniform", name))
            {
                var _shape = Tensorflow.tf.constant(shape.Select(x => (long)x).ToArray());
                // TODO: TensorFlow.NET - minval, maxval support...
                //return Out(gen_random_ops.random_uniform(_shape, _dtype, seed, name: "uniform"));
                return Out(gen_math_ops.add(
                    gen_math_ops.mul(
                        gen_random_ops.random_uniform(_shape, _dtype, seed, name: "uniform"),
                        Tensorflow.tf.constant(maxval - minval, dtype: _dtype)
                    ),
                    Tensorflow.tf.constant(minval, dtype: _dtype),
                    name: "scaled"
                ));
            }
        }

        public KerasSharp.Engine.Topology.Tensor relu(KerasSharp.Engine.Topology.Tensor x)
        {
            return Out(tf.Relu(In(x)));
        }

        public KerasSharp.Engine.Topology.Tensor sigmoid(KerasSharp.Engine.Topology.Tensor x)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2817
            return Out(tf.Sigmoid(In(x)));
        }

        public KerasSharp.Engine.Topology.Tensor softmax(KerasSharp.Engine.Topology.Tensor x)
        {
            return Out(tf.Softmax(In(x).output));
        }

        public KerasSharp.Engine.Topology.Tensor softplus(KerasSharp.Engine.Topology.Tensor x)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor softsign(KerasSharp.Engine.Topology.Tensor x)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor sqrt(KerasSharp.Engine.Topology.Tensor x)
        {
            return Out(tf.Sqrt(In(x)));
        }

        public KerasSharp.Engine.Topology.Tensor pow(KerasSharp.Engine.Topology.Tensor x, KerasSharp.Engine.Topology.Tensor p, string name = null)
        {
            return Out(gen_math_ops.pow(In(x).tensor, In(p).tensor, name: name));
        }

        public KerasSharp.Engine.Topology.Tensor square(KerasSharp.Engine.Topology.Tensor w)
        {
            return Out(tf.Square(In(w)));
        }

        public KerasSharp.Engine.Topology.Tensor sum(KerasSharp.Engine.Topology.Tensor x, int[] axis, bool keepdims = false, string name = null)
        {
            return Out(math_ops.reduce_sum(In(x).tensor, Tensorflow.tf.constant(axis)));
        }

        public KerasSharp.Engine.Topology.Tensor sum(KerasSharp.Engine.Topology.Tensor x, int axis, bool keepdims = false, string name = null)
        {
            return Out(math_ops.reduce_sum(In(x).tensor, Tensorflow.tf.constant(axis)));
        }

        public KerasSharp.Engine.Topology.Tensor sum(List<KerasSharp.Engine.Topology.Tensor> x, int[] axis = null, bool keepdims = false, string name = null)
        {
            throw new NotImplementedException();
        }

        public object sum(KerasSharp.Engine.Topology.Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor sum(double v, KerasSharp.Engine.Topology.Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor tanh(KerasSharp.Engine.Topology.Tensor x)
        {
            return Out(tf.Tanh(In(x)));
        }

        public KerasSharp.Engine.Topology.Tensor truncated_normal(int[] shape, double v, double stddev, DataType? dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor truncated_normal(int?[] shape, double v, double stddev, DataType? dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor update(KerasSharp.Engine.Topology.Tensor x, KerasSharp.Engine.Topology.Tensor new_x, string name = null)
        {
            TensorFlowTensor _x = In(x);
            return Out(tf.Assign(_x.output, In(new_x), operName: name));
        }

        public KerasSharp.Engine.Topology.Tensor update_add<T>(KerasSharp.Engine.Topology.Tensor x, T increment, string name = null)
            where T : struct
        {
            TensorFlowTensor _x = In(x);
            return Out(tf.AssignAdd(_x, _constant(increment), operName: name));
        }

        public KerasSharp.Engine.Topology.Tensor print_tensor(KerasSharp.Engine.Topology.Tensor x, string message)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2204
            TensorFlowTensor _x = In(x);
            return Out(tf.Print(_x, new[] { _x.output }, message));
        }

        /// <summary>
        ///   Instantiates a variable and returns it.
        /// </summary>
        /// 
        /// <param name="value">C# array, initial value of the tensor.</param>
        /// <param name="dtype">Tensor type.</param>
        /// <param name="name">Optional name string for the tensor.</param>
        /// 
        public KerasSharp.Engine.Topology.Tensor variable<T>(T value, DataType? dtype = null, string name = null)
            where T : struct
        {
            if (dtype == null)
                dtype = floatx();

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L308
            var _dtype = In(dtype.Value);

            using (var scope = name_scope("Variable", name))
            {
                var t = new TensorFlowTensor(this);
                t.output = tf.VariableV2(TFShape.Scalar, _dtype, operName: "var");
                var init = _constant(value, _dtype, operName: "init");
                init = tf.Print(init, new[] { init }, $"initializing {scope.Name}");
                tf.AddInitVariable(tf.Assign(t.output, init, operName: "assign").Operation);
                t._keras_shape = new int?[] { };
                t._uses_learning_phase = false;
                return t;
            }
        }

        /// <summary>
        ///   Instantiates a variable and returns it.
        /// </summary>
        /// 
        /// <param name="array">C# array, initial value of the tensor.</param>
        /// <param name="dtype">Tensor type.</param>
        /// <param name="name">Optional name string for the tensor.</param>
        /// 
        public KerasSharp.Engine.Topology.Tensor variable(Array array, DataType? dtype = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L308
            var _dtype = In(dtype.Value);

            var t = new TensorFlowTensor(this);
            t.output = tf.VariableV2(In(array.GetLength()), _dtype, operName: name);

            string varName = t.output.Operation.Name;

            var init = _constant(array, _dtype, operName: $"{varName}/init");
            init = tf.Print(init, new[] { init }, $"initializing {varName}");
            tf.AddInitVariable(tf.Assign(t.output, init, operName: $"{varName}/assign").Operation);
            t._keras_shape = array.GetLength().Apply(x => (int?)x);
            t._uses_learning_phase = false;
            return t;
        }

        /// <summary>
        ///   Instantiates a variable and returns it.
        /// </summary>
        /// 
        /// <param name="tensor">Tensor, initial value of the tensor.</param>
        /// <param name="dtype">Tensor type.</param>
        /// <param name="name">Optional name string for the tensor.</param>
        /// 
        public KerasSharp.Engine.Topology.Tensor variable(KerasSharp.Engine.Topology.Tensor tensor, DataType? dtype = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            var _tensor = In(tensor);
            var _dtype = In(dtype.Value);
            TFShape _shape = In(tensor.shape);

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L308

            var t = new TensorFlowTensor(this);

            t.output = tf.VariableV2(_shape, _dtype, operName: name);

            string varName = t.output.Operation.Name;

            TF_Output init;
            if (_tensor.tensor == null)
                init = _tensor.output;
            else
                init = Tensorflow.tf.cast(Tensorflow.tf.constant(_tensor.tensor), _dtype, name: $"{varName}/init")._as_tf_output();

            init = tf.Print(init, new[] { init }, $"initializing {varName}");
            tf.AddInitVariable(tf.Assign(t.output, init, operName: $"{varName}/assign").Operation);
            t._keras_shape = tensor.shape;
            t._uses_learning_phase = false;
            return t;
        }

        public KerasSharp.Engine.Topology.Tensor transpose(KerasSharp.Engine.Topology.Tensor tensor)
        {
            return Out(tf.Transpose(In(tensor).output));
        }

        public KerasSharp.Engine.Topology.Tensor transpose(KerasSharp.Engine.Topology.Tensor tensor, int[] perm)
        {
            return Out(tf.Transpose(In(tensor).output, _constant(perm)));
        }


        public object eval(KerasSharp.Engine.Topology.Tensor tensor)
        {
            var _tensor = In(tensor);
            return eval(_tensor.output);
        }

        public object eval(TF_Output output)
        {
            try
            {
                // Initialize variables if necessary
                //TFOperation[] ops = tf.GetGlobalVariablesInitializer();
                //if (ops.Length > 0)
                //    _SESSION.Run(new TF_Output[] { }, new Tensorflow.Tensor[] { }, new TF_Output[] { }, ops);
                
                var init = Tensorflow.tf.global_variables_initializer();
                if (init.NumOutputs > 0)
                    _SESSION.run(init);
            }
            catch
            {
                // temporary workaround until changes are sent to TensorFlowSharp
            }

            // Evaluate tensor
            Tensorflow.Tensor[] result = _SESSION.Run(new TF_Output[] { }, new Tensorflow.Tensor[] { }, new[] { output });

            if (result.Length == 1)
                return result[0].GetValue();

            return result.Apply(x => x.GetValue());
        }



        public KerasSharp.Engine.Topology.Tensor conv1d(KerasSharp.Engine.Topology.Tensor inputs, KerasSharp.Engine.Topology.Tensor kernel, int strides, PaddingType padding, DataFormatType? data_format = null, int dilation_rate = 1, string name = null)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.Engine.Topology.Tensor conv2d(KerasSharp.Engine.Topology.Tensor inputs, KerasSharp.Engine.Topology.Tensor kernel, int[] strides, PaddingType padding, DataFormatType? data_format = null, int[] dilation_rate = null, string name = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L3102
            if (data_format == null)
                data_format = image_data_format();

            if (dilation_rate == null)
                dilation_rate = new int[] { 1, 1 };

            if (!dilation_rate.IsEqual(new[] { 1, 1 }))
                throw new NotImplementedException();

            TF_Output x = In(inputs).output;
            TF_Output _kernel = In(kernel).output;

            // With 4d inputs, tf.nn.convolution only supports
            // data_format NHWC, so we transpose the inputs
            // in case we are in data_format channels_first.
            x = _preprocess_conv2d_input(x, data_format.Value);
            string _padding = _preprocess_padding(padding);
            x = tf.Conv2D(
                input: x,
                filter: _kernel,
                //dilation_rate: dilation_rate,
                strides: strides.Select(i => (long)i).ToArray(),
                padding: _padding,
                data_format: "NHWC");
            return Out(_postprocess_conv2d_output(x, data_format.Value));
        }

        /// <summary>
        ///   Transpose and cast the output from conv2d if needed.
        /// </summary>
        private TF_Output _postprocess_conv2d_output(TF_Output x, DataFormatType data_format)
        {
            if (data_format == DataFormatType.ChannelsFirst)
                x = tf.Transpose(x, _constant(new[] { 0, 3, 1, 2 }));

            if (floatx() == DataType.DtDouble)
                x = Tensorflow.tf.cast(x, TF_DataType.TF_DOUBLE);
            return x;
        }

        /// <summary>
        ///   Convert keras' padding to tensorflow's padding.
        /// </summary>
        /// 
        public string _preprocess_padding(PaddingType padding)
        {
            switch (padding)
            {
                case PaddingType.Same:
                    return "SAME";
                case PaddingType.Valid:
                    return "VALID";
            }

            throw new ArgumentException($"Invalid padding: {padding}");
        }

        /// <summary>
        ///   Transpose and cast the input before the conv2d.
        /// </summary>
        private TF_Output _preprocess_conv2d_input(TF_Output x, DataFormatType data_format)
        {
            if (x.OutputType == TF_DataType.TF_DOUBLE)
                x = Tensorflow.tf.cast(x, TF_DataType.TF_FLOAT);

            if (data_format == DataFormatType.ChannelsFirst)
            {
                // TF uses the last dimension as channel dimension,
                // instead of the 2nd one.
                // TH input shape: (samples, input_depth, rows, cols)
                // TF input shape: (samples, rows, cols, input_depth)
                x = tf.Transpose(x, _constant(new[] { 0, 2, 3, 1 }));
            }

            return x;
        }

        public KerasSharp.Engine.Topology.Tensor conv3d(KerasSharp.Engine.Topology.Tensor inputs, KerasSharp.Engine.Topology.Tensor kernel, int[] strides, PaddingType padding, DataFormatType? data_format = null, int[] dilation_rate = null, string name = null)
        {
            throw new NotImplementedException();
        }



        /// <summary>
        ///   Instantiates an all-zeros variable and returns it.
        /// </summary>
        /// <param name="shape">Tuple of integers, shape of returned Keras variable.</param>
        /// <param name="dtype">Data type of returned Keras variable.</param>
        /// <param name="name">String, name of returned Keras variable.</param>
        /// <returns>A variable(including Keras metadata), filled with <c>0.0</c>.</returns>
        public KerasSharp.Engine.Topology.Tensor zeros(int?[] shape, DataType? dtype = null, string name = null)
        {
            return zeros(shape.Select(i => i.Value).ToArray(), dtype, name);
        }

        /// <summary>
        ///   Instantiates an all-zeros variable and returns it.
        /// </summary>
        /// <param name="shape">Tuple of integers, shape of returned Keras variable.</param>
        /// <param name="dtype">Data type of returned Keras variable.</param>
        /// <param name="name">String, name of returned Keras variable.</param>
        /// <returns>A variable(including Keras metadata), filled with <c>0.0</c>.</returns>
        public KerasSharp.Engine.Topology.Tensor zeros(int[] shape, DataType? dtype = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            // The following is not necessary since C# is strongly typed:
            // shape = tuple(map(int, shape))
            // tf_dtype = _convert_string_dtype(dtype)

            // However, we might have to perform other conversions of our own:
            Type type = Tensorflow.Tensor.TypeFromTensorType(In(dtype.Value));
            Array zeros = Array.CreateInstance(type, shape);

            return this.variable(array: zeros, name: name);
        }

        /// <summary>
        ///   Element-wise equality between two tensors.
        /// </summary>
        /// 
        /// <param name="x">Tensor or variable.</param>
        /// <param name="y">Tensor or variable.</param>
        /// 
        /// <returns>A bool tensor.</returns>
        /// 
        public KerasSharp.Engine.Topology.Tensor equal(KerasSharp.Engine.Topology.Tensor x, KerasSharp.Engine.Topology.Tensor y)
        {
            return Out(tf.Equal(In(x), In(y)));
        }

        /// <summary>
        ///   Returns the index of the maximum value along an axis.
        /// </summary>
        /// 
        /// <param name="x">Tensor or variable.</param>
        /// <param name="axis">The axis along which to perform the reduction.</param>
        /// 
        /// <returns>A tensor.</returns>
        /// 
        public KerasSharp.Engine.Topology.Tensor argmax(KerasSharp.Engine.Topology.Tensor x, int axis = -1)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L1332
            //axis = _normalize_axis(axis, ndim(x));
            return Out(tf.ArgMax(In(x), Tensorflow.tf.constant(axis)));
        }

        public KerasSharp.Engine.Topology.Tensor round(KerasSharp.Engine.Topology.Tensor x)
        {
            return Out(tf.Round(In(x)));
        }

        public DataType floatx()
        {
            return DataType.DtFloat;
        }


        public string MakeName(string operName, string userName)
        {
            if (userName == null)
            {
                var k = tf.CurrentNameScope == "" ? operName : tf.CurrentNameScope + "/" + operName;
                return $"{k}_{str(get_uid(k))}";
            }

            if (tf.CurrentNameScope == "")
                return userName;
            return tf.CurrentNameScope + "/" + userName;
        }



        #region conversion

        public TFShape In(int?[] shape)
        {
            return new TFShape(shape.Select(x => x.HasValue ? (long)x.Value : -1).ToArray());
        }

        public TFShape In(int[] shape)
        {
            return new TFShape(shape.Select(x => (long)x).ToArray());
        }

        public KerasSharp.Engine.Topology.Tensor Out(TF_Output output, string name = null)
        {
            if (name != null)
                output = tf.Identity(output, operName: name);

            return new TensorFlowTensor(this)
            {
                output = output
            };
        }

        public KerasSharp.Engine.Topology.Tensor Out(Tensorflow.Tensor output)
        {
            return Out(Tensorflow.tf.constant(output));
        }

        public TensorFlowTensor In(KerasSharp.Engine.Topology.Tensor output)
        {
            return (TensorFlowTensor)output;
        }

        public TensorFlowTensor In(TF_Output output)
        {
            return new TensorFlowTensor(this) { output = output };
        }

        public static TF_DataType In(DataType dataType)
        {
            return (TF_DataType)dataType;
        }

        public static TF_DataType? In(DataType? dataType)
        {
            if (dataType == null)
                return null;
            return (TF_DataType)dataType.Value;
        }

        public static DataType? Out(TF_DataType? dataType)
        {
            if (dataType == null)
                return null;
            return Out(dataType.Value);
        }

        public static DataType Out(TF_DataType dataType)
        {
            if ((int)dataType > 100)
                return (DataType)((dataType - 100));
            return (DataType)dataType;
        }

        #endregion





        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                    if (tf != null)
                        tf.Dispose();
                    if (_SESSION != null)
                        _SESSION.Dispose();
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.

                disposedValue = true;
                tf = null;
                _SESSION = null;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        // ~TensorFlowBackend() {
        //   // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
        //   Dispose(false);
        // }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // TODO: uncomment the following line if the finalizer is overridden above.
            // GC.SuppressFinalize(this);
        }
        #endregion
    }
}
