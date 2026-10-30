<script setup lang="ts">
import { models, traceLengths } from '../lib/config';
import { useModelStore } from '../stores/model';

const model = useModelStore();
</script>

<template>
  <section class="rail-section">
    <h2>Model</h2>
    <div class="field">
      <label for="model-sel">Model <span class="label-hint">(downloaded once, then cached)</span></label>
      <select id="model-sel" v-model="model.modelId" :disabled="model.loading">
        <option v-for="m in models" :key="m.id" :value="m.id">{{ m.label }}</option>
      </select>
    </div>
    <div class="field">
      <label for="trace-len-sel">Longest traceable conversation</label>
      <select id="trace-len-sel" v-model.number="model.traceLength" :disabled="model.loading"
              title="Longer traces make every generated token slower and take more memory; changing it reloads the model">
        <option v-for="n in traceLengths" :key="n" :value="n">{{ n }} tokens</option>
      </select>
    </div>
    <button class="wide" :disabled="!model.canLoad" @click="model.load()">
      {{ model.loading ? 'Loading…' : !model.canLoad && model.ready ? 'Loaded' : model.ready && model.modelId === model.loadedModel ? 'Reload model' : 'Load model' }}
    </button>
    <div v-if="model.progressVisible" class="progress-wrap">
      <div class="progress-status" :class="{ error: model.statusLevel === 'error', warn: model.statusLevel === 'warn' }">
        {{ model.status }}
      </div>
      <div class="progress-files">
        <div v-for="row in model.progressRows" :key="row.key" class="progress-file">
          <div class="progress-bar"><div class="progress-fill" :style="{ width: `${row.pct}%` }"></div></div>
          <div class="progress-label">{{ row.label }}</div>
        </div>
      </div>
    </div>
  </section>
</template>
