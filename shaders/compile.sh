#!/usr/bin/env bash

$HOME/vulkan-sdk/1.1.130.0/x86_64/bin/glslc shader.vert -o vert.spv
$HOME/vulkan-sdk/1.1.130.0/x86_64/bin/glslc shader.frag -o frag.spv