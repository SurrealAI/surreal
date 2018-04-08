variable "zone" {
  default = "us-west1-b"
}

variable "project" {
  default = "surreal-dev-188523"
}

variable "credential" {
  default = "surreal.json"
}

variable "cluster_name" {
  default = "kurreal"
}

provider "google" {
  credentials = "${file(${var.credential})}"
  project     = "${var.project}"
  region      = "us-west1-b"
}

resource "google_container_cluster" "kurreal" {
  name = "${var.cluster_name}"
  zone = "${var.zone}"
  min_master_version = "1.9"

  initial_node_count = 1
  remove_default_node_pool = true
}

# Default pool for kubernetes system pods and miscellaneous services
resource "google_container_node_pool" "n1-standard-1" {
  name    = "n1-standard-1"
  zone    = "${google_container_cluster.kurreal.zone}"
  cluster = "${google_container_cluster.kurreal.name}"

  autoscaling {
    min_node_count   = 0
    max_node_count   = 500
  }
  initial_node_count = 3

  node_config {
    machine_type = "n1-standard-1"
    labels {
      surreal-node  = "n1-standard-1"
      surreal-alias = "misc"
    }
  }

}

# This small pool is usually used for agents
resource "google_container_node_pool" "n1-standard-2" {
  name    = "n1-standard-2"
  zone    = "${google_container_cluster.kurreal.zone}"
  cluster = "${google_container_cluster.kurreal.name}"

  autoscaling {
    min_node_count   = 0
    max_node_count   = 500
  }
  initial_node_count = 1

  node_config {
    machine_type = "n1-standard-2"
    disk_size_gb = 30
    labels {
      surreal-machine  = "n1-standard-2"
      surreal-node = "agent"
    }
    taint {
      key = "surreal"
      value = "true"
      effect = "NO_EXECUTE"
    }
  }
}

# This machine can be used for cpu-based-learner
resource "google_container_node_pool" "n1-highmem-8" {
  name    = "n1-highmem-8"
  zone    = "${google_container_cluster.kurreal.zone}"
  cluster = "${google_container_cluster.kurreal.name}"

  autoscaling {
    min_node_count   = 0
    max_node_count   = 100
  }
  initial_node_count = 1

  node_config {
    machine_type = "n1-highmem-8"
    labels {
      surreal-machine  = "n1-highmem-8"
      surreal-node = "nonagent-cpu"
    }
    taint {
      key = "surreal"
      value = "true"
      effect = "NO_EXECUTE"
    }
  }
}

# This machine can be used for gpu-based-learner
resource "google_container_node_pool" "n1-highmem-8-1k80" {
  name    = "n1-highmem-8-1k80"
  zone    = "${google_container_cluster.kurreal.zone}"
  cluster = "${google_container_cluster.kurreal.name}"

  autoscaling {
    min_node_count   = 0
    max_node_count   = 100
  }
  initial_node_count = 1

  node_config {
    machine_type = "n1-highmem-8"
    guest_accelerator {
      type  = "nvidia-tesla-k80"
      count = 1
    }
    labels {
      surreal-machine  = "n1-highmem-8-1k80"
      surreal-node = "nonagent-gpu"
    }
    taint {
      key = "surreal"
      value = "true"
      effect = "NO_EXECUTE"
    }
  }
}


# This machine can be used for gpu-based-learner
resource "google_container_node_pool" "n1-standard-16-2k80" {
  name    = "n1-standard-16-2k80"
  zone    = "${google_container_cluster.kurreal.zone}"
  cluster = "${google_container_cluster.kurreal.name}"

  autoscaling {
    min_node_count   = 0
    max_node_count   = 100
  }
  initial_node_count = 1

  node_config {
    machine_type = "n1-standard-16"
    guest_accelerator {
      type  = "nvidia-tesla-k80"
      count = 2
    }
    labels {
      surreal-machine  = "n1-standard-16-2k80"
      surreal-node = "nonagent-2k80"
    }
    taint {
      key = "surreal"
      value = "true"
      effect = "NO_EXECUTE"
    }
  }
}


# This machine can be used for gpu-based-learner
resource "google_container_node_pool" "n1-standard-16-1p100" {
  name    = "n1-standard-16-1p100"
  zone    = "${google_container_cluster.kurreal.zone}"
  cluster = "${google_container_cluster.kurreal.name}"

  autoscaling {
    min_node_count   = 0
    max_node_count   = 100
  }
  initial_node_count = 1

  node_config {
    machine_type = "n1-standard-16"
    guest_accelerator {
      type  = "nvidia-tesla-p100"
      count = 1
    }
    labels {
      surreal-machine  = "n1-standard-16-1p100"
      surreal-node = "nonagent-gpu"
    }
    taint {
      key = "surreal"
      value = "true"
      effect = "NO_EXECUTE"
    }
  }
}


