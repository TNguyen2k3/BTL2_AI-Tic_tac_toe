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

    public Vector2Int PredictMove(float[] board) // board: 81 phần tử, giá trị 1, -1, 0
    {
        float[] inputData = new float[2 * 9 * 9]; // 2 channels

        for (int i = 0; i < 81; i++)
        {
            if (board[i] == 1)
                inputData[i] = 1f; // channel 0: bot
            else if (board[i] == -1)
                inputData[81 + i] = 1f; // channel 1: opponent
        }

        Tensor input = new Tensor(1, 9, 9, 2, inputData); // shape: (1, height, width, channels)

        worker.Execute(input);
        Tensor output = worker.PeekOutput(); // (1, 81)

        int bestIndex = -1;
        float bestScore = float.MinValue;

        for (int i = 0; i < 81; i++)
        {
            if (board[i] != 0) continue; // bỏ qua ô đã đánh
            float score = output[0, i];
            if (score > bestScore)
            {
                bestScore = score;
                bestIndex = i;
            }
        }

        input.Dispose();
        output.Dispose();

        if (bestIndex == -1)
            return new Vector2Int(-1, -1); // fallback nếu không có ô hợp lệ

        return new Vector2Int(bestIndex / 9, bestIndex % 9);
    }

    public void Dispose()
    {
        worker?.Dispose();
    }
}
