import * as tfvis from "@tensorflow/tfjs-vis";
const statusElement = document.getElementById("status");
export function updateStatus(message) {
  statusElement.innerText = message;
}