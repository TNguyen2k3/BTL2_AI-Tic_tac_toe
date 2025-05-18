using Unity.Barracuda;
using UnityEngine;
[System.Serializable]
public class DQNBot
{
    private Model model;
    private IWorker worker;

    public DQNBot(NNModel modelAsset)
    {
        model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
    }

    public Vector2Int PredictMove(float[] board)
    {
        Tensor input = new Tensor(1, 81, board);
        worker.Execute(input);
        Tensor output = worker.PeekOutput();

        int bestIndex = 0;
        float bestScore = float.MinValue;

        for (int i = 0; i < output.length; i++)
        {
            if (board[i] != 0) continue;
            if (output[i] > bestScore)
            {
                bestScore = output[i];
                bestIndex = i;
            }
        }

        input.Dispose();
        output.Dispose();

        return new Vector2Int(bestIndex / 9, bestIndex % 9);
    }

    public void Dispose()
    {
        worker?.Dispose();
    }
}
