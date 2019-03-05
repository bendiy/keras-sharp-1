using KerasSharp.Engine.Topology;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;

using static KerasSharp.Python;

namespace KerasSharp
{
    public static class TensorFlowSharpEx
    {


        // Returns range(0, rank(x)) if reduction_indices is null
        public static TF_Output ReduceDims(this Graph g, TF_Output input, TF_Output? axis = null)
        {
            if (axis.HasValue)
                return axis.Value;

            // Fast path: avoid creating Rank and Range ops if ndims is known.
            long[] shape = g.GetTensorShape(input).ToArray();
            // TODO: TensorFlow.NET???
            //long[] shape = { };
            //var s = new Status();
            //c_api.TF_GraphGetTensorShape(g, input, shape, 1, s);
            if (shape.Length >= 0)
            {
                // The python code distinguishes between tensor and sparsetensor

                var array = new int[shape.Length];
                for (int i = 0; i < array.Length; i++)
                    array[i] = i;

                return tf.constant(array, TF_DataType.TF_INT32)._as_tf_output();
            }

            return gen_math_ops.range(tf.constant(0), tf.constant(shape.Length), tf.constant(1))._as_tf_output();
        }

        #region Staging area - remove after those operations have been implemented in TensorFlowSharp

        public static TF_Output Transpose(this Graph g, TF_Output a, TF_Output? perm = null, string operName = null)
        {
            throw new NotImplementedException("https://github.com/migueldeicaza/TensorFlowSharp/pull/178");
        }

        public static TF_Output Cond(this Graph g, TF_Output pred, Func<TF_Output> true_fn, Func<TF_Output> false_fn, string operName = null)
        {
            throw new NotImplementedException("https://github.com/migueldeicaza/TensorFlowSharp/pull/176");
        }

        #endregion
    }
}
