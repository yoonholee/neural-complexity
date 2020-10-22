import os
import pickle

from .sine import get_sine_loader


def get_loader(task, batch_size, num_steps):
    if task == "sine":
        loader = get_sine_loader(batch_size=batch_size, num_steps=num_steps)
    else:
        raise ValueError(f"task={task} is not implemented")
    return loader


def get_task(saved, task, batch_size, num_steps):
    if not saved:
        return get_loader(task, batch_size, num_steps)

    os.makedirs("data/saved", exist_ok=True)
    filename = f"data/saved/{task}_{batch_size}_{num_steps}.pkl"

    if os.path.exists(filename):
        with open(filename, "rb") as handle:
            tasks = pickle.load(handle)
    else:
        test_task_gen = get_loader(
            task=task, batch_size=batch_size, num_steps=num_steps
        )
        tasks = [t for t in test_task_gen]
        with open(filename, "wb") as handle:
            pickle.dump(tasks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tasks
