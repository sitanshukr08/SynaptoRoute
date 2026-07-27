name: Bug Report
description: Create a report to help us reproduce and fix a bug
title: "[BUG] "
labels: ["bug"]
body:
  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: A clear and concise description of what the bug is.
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Reproduction Code
      description: Provide a minimal code snippet or script to reproduce the issue.
      placeholder: |
        from synaptoroute import AdaptiveRouter, Route
        router = AdaptiveRouter()
        ...
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment Metadata
      description: Output of `synaptoroute info` or Python/OS details.
    validations:
      required: false
