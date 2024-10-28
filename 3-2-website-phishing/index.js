import * as tf from "@tensorflow/tfjs";
import { WebsitePhishingDataset } from "./data";
import * as ui from "./ui"

const data = new WebsitePhishingDataset();
data.loadData().then(async () => {
  await ui.updateStatus("Getting training and testing data...");
})