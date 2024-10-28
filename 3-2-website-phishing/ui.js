const statusElement = document.getElementById("status");
export function updateStatus(message) {
  statusElement.innerText = message;
}