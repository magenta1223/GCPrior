import pickle as pkl
from pathlib import Path
from typing import (Dict, Iterable, List, Optional, Sequence, Tuple, Union,
                    overload)

import cv2
import fire
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from carla_env.dataset import Dataset

ROUTE_IMAGE = "route_image/Route_point.png"


@overload
def draw_path(
    dataset: Dataset,
    output_filepath: None = None,
    background: Optional[Union[cv2.Mat, np.ndarray]] = None,
) -> cv2.Mat:
    ...

@overload
def draw_path(
    dataset: Dataset,
    output_filepath: Union[str, Path],
    background: Optional[Union[cv2.Mat, np.ndarray]] = None,
) -> None:
    ...


def draw_path(
    dataset: Dataset,
    output_filepath: Optional[Union[str, Path]] = None,
    background: Optional[Union[cv2.Mat, np.ndarray]] = None,
):
    """Draw path of the dataset on the route image.

    The start point is a green circle, the end point is a purple circle, and the path is
    a blue line. The target location is a red circle.

    Args:
        dataset (Dataset): Dataset

    Returns:
        cv2.Mat: Image if output_filepath is None, otherwise None
    
    """
    # Read background image
    if background is not None:
        image = background
    else:
        image = cv2.imread(ROUTE_IMAGE)

    # Read dataset
    observations = dataset["observations"]
    sensor = observations["sensor"]
    lidar_bin = dataset.get("lidar_bin", 80)
    offset = lidar_bin + 9

    def transform(p: np.ndarray):
        """
        Transform from sensor coordinate to image coordinate
        Sensor coordinate: X - [-125, 120], Y - [-73, 146]
        Image coordinate: X - [0, 1472], Y - [0, 1321]
        Rotation: 0.0125rad, counterclockwise

        """

        # translation
        p[:, 0], p[:, 1] = (
            (p[:, 0] + 125) * 1472 / 245,
            (p[:, 1] + 73) * 1321 / 219,
        )

        # rotation
        p[:, 0], p[:, 1] = (
            p[:, 0] * np.cos(-0.0125) - p[:, 1] * np.sin(-0.0125),
            p[:, 0] * np.sin(-0.0125) + p[:, 1] * np.cos(-0.0125),
        )

        return p

    # Draw path
    path = transform(
        sensor[:, offset:(offset + 2)]
    ).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [path], False, (255, 70, 70), 5, lineType=cv2.LINE_AA)

    # Draw start and end point
    start = path[0].reshape(-1)
    end = path[-1].reshape(-1)

    cv2.circle(image, tuple(start), 12, (70, 255, 70), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, tuple(end), 12, (128, 0, 128), -1, lineType=cv2.LINE_AA)

    # Draw target location
    target = transform(
        sensor[-1, (offset + 12):(offset + 14)][None, :]
    ).astype(np.int32).reshape(-1)
    cv2.circle(image, tuple(target), 12, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    # Put text of the reason of done
    info = dataset["infos"][-1]
    reason = ""
    for key, value in info.items():
        if key.startswith("done_") and value:
            reason = key[5:]
            break

    if reason:
        cv2.putText(
            image,
            reason,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    if output_filepath is not None:
        # Save image
        print(f"Saving image to {output_filepath} ...")
        cv2.imwrite(str(output_filepath), image)
    else:
        return image


def plot_action_distribution(
    *datasets: Sequence[Dataset],
    labels: Optional[Sequence[str]] = None,
    output_filepath: Optional[Union[str, Path]] = None,
):
    """Plot the action distribution of the dataset as a frequency polygon.
    
    The action distribution is plotted as a frequency polygon, which is a line graph
    that displays the distribution of a continuous variable. The x-axis represents the
    action value and the y-axis represents the frequency of the action value.

    It plots to the current axes if `output_filepath` is None. Otherwise, it saves the
    figure to the `output_filepath`.
    
    Args:
        datasets (Dataset): Datasets
        labels (Iterable[str]): Labels of the datasets
        output_filepath (str | Path): Output filepath
    
    """
    if isinstance(output_filepath, Path):
        output_filepath = str(output_filepath)

    if labels is None:
        labels = [f"Dataset {i}" for i in range(len(datasets))]

    if len(datasets) != len(labels):
        raise ValueError(
            f"Length of datasets ({len(datasets)}) and labels ({len(labels)}) do not "
            f"match."
        )

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    freq: Dict[str, Dict[str, np.ndarray]] = {
        "throttle": {}, "steering": {}, "brake": {}
    }

    for dataset, label in zip(datasets, labels):
        actions = np.concatenate(list(data["actions"] for data in tqdm(
            dataset, desc=f"Loading {label} dataset"
        )), axis=0)
        freq["throttle"][label] = actions[:, 0]
        freq["steering"][label] = actions[:, 1]
        freq["brake"][label] = actions[:, 2]

    print("Plotting ...")
    for i, y_lim in enumerate((15, 5, 0.5, 0.05)):
        for j, (key, value) in enumerate(freq.items()):
            ax = axes[i][j]    # type: ignore
            ax.set_title(key.capitalize())
            ax.set_xlabel("Action")
            ax.set_ylabel("Frequency")

            x_min = min(v.min() for v in value.values())
            x_max = max(v.max() for v in value.values())
            margin = (x_max - x_min) * 0.1
            ax.set_xlim(x_min - margin, x_max + margin)

            ax.set_ylim(0, y_lim)
            ax.grid(True)

            sns.kdeplot(data=value, ax=ax, fill=True, common_norm=False, alpha=0.5)

    if output_filepath is None:
        plt.show()
    else:
        fig.savefig(output_filepath)


class DatasetSequence(Sequence[Dataset]):
    def __init__(self, datasets: Iterable[Union[str, Path]]):
        self.__datasets: List[str] = []
        for filename in tqdm(datasets, desc="Validating datasets"):
            try:
                with open(filename, "rb") as f:
                    pkl.load(f)
            except pkl.UnpicklingError as e:
                print(f"Failed to load {filename}: {e}")
            else:
                self.__datasets.append(str(filename))

    def __len__(self):
        return len(self.__datasets)

    def __getitem__(self, idx: int) -> Dataset:
        with open(self.__datasets[idx], "rb") as f:
            return pkl.load(f)


class Program:
    def draw_path(self, src: str, dst: Optional[str] = None):
        self.__src = Path(src)
        if self.__src.is_file() and dst is None:
            raise ValueError("dst must be specified when src is a file.")
        self.__dst = Path(dst) if dst is not None else self.__src
        if self.__dst.is_file():
            raise ValueError("dst must be a directory.")

        if self.__src.is_dir():
            for filename in self.__src.glob("*.pkl"):
                try:
                    with open(filename, "rb") as f:
                        dataset = pkl.load(f)
                except pkl.UnpicklingError as e:
                    print(f"Failed to load {filename}: {e}")
                    continue
                draw_path(dataset, output_filepath=self.__dst / f"{filename.stem}.png")
        else:
            with open(self.__src, "rb") as f:
                dataset = pkl.load(f)
            draw_path(dataset, output_filepath=self.__dst / f"{self.__src.stem}.png")

    def plot_action_distribution(
        self,
        *srcs: str,
        dst: Optional[str] = None,
    ):
        if len(srcs) % 2 != 0:
            raise ValueError("srcs must be a list of (dataset, label) pairs.")

        if dst is not None:
            self.__dst = Path(dst)
            if self.__dst.is_dir():
                raise ValueError("dst must be a file.")
        else:
            self.__dst = None

        self.__srcs: List[DatasetSequence] = []
        self.__labels: List[str] = []
        for i in range(0, len(srcs), 2):
            label = srcs[i]
            pth = Path(srcs[i + 1])
            dataset = DatasetSequence(pth.glob("**/*.pkl"))

            print(f"Loaded {len(dataset)} datasets for {label} label.")

            self.__srcs.append(dataset)
            self.__labels.append(label)

        plot_action_distribution(
            *self.__srcs,
            labels=self.__labels,
            output_filepath=self.__dst,
        )


if __name__ == "__main__":
    fire.Fire(Program)
