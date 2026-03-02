<template>
  <div v-if="hasAny" class="conversation-sources-panel">
    <button class="panel-header" type="button" @click="toggleExpanded">
      <div class="header-left">
        <div class="title">来源</div>
        <div class="summary">
          <span v-if="knowledgeChunks.length > 0">知识库 {{ knowledgeChunks.length }}</span>
          <span v-if="webSources.length > 0">网络 {{ webSources.length }}</span>
        </div>
      </div>
      <ChevronDown :size="14" class="expand-icon" :class="{ rotated: !expanded }" />
    </button>

    <div v-if="expanded" class="panel-body">
      <KnowledgeSourceSection v-if="knowledgeChunks.length > 0" :chunks="knowledgeChunks" />
      <WebSearchSourceSection v-if="webSources.length > 0" :sources="webSources" />
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { ChevronDown } from 'lucide-vue-next'
import KnowledgeSourceSection from '@/components/KnowledgeSourceSection.vue'
import WebSearchSourceSection from '@/components/WebSearchSourceSection.vue'

const props = defineProps({
  sources: {
    type: Object,
    default: () => ({})
  },
  knowledgeChunks: {
    type: Array,
    default: () => []
  },
  webSources: {
    type: Array,
    default: () => []
  }
})

const expanded = ref(false)
const knowledgeChunks = computed(() =>
  Array.isArray(props.sources?.knowledgeChunks) ? props.sources.knowledgeChunks : props.knowledgeChunks
)
const webSources = computed(() =>
  Array.isArray(props.sources?.webSources) ? props.sources.webSources : props.webSources
)

const hasAny = computed(
  () =>
    (Array.isArray(knowledgeChunks.value) && knowledgeChunks.value.length > 0) ||
    (Array.isArray(webSources.value) && webSources.value.length > 0)
)

const toggleExpanded = () => {
  expanded.value = !expanded.value
}
</script>

<style scoped lang="less">
.conversation-sources-panel {
  margin: 8px 0 14px 0;
  border: 1px solid var(--gray-200);
  border-radius: 8px;
  background: var(--gray-25);
  overflow: hidden;

  .panel-header {
    width: 100%;
    border: none;
    background: transparent;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    cursor: pointer;

    .header-left {
      display: flex;
      align-items: baseline;
      gap: 10px;
      min-width: 0;

      .title {
        font-size: 13px;
        font-weight: 600;
        color: var(--gray-700);
        line-height: 20px;
      }

      .summary {
        display: flex;
        align-items: baseline;
        gap: 8px;
        font-size: 12px;
        color: var(--gray-600);
        line-height: 20px;
      }
    }

    .expand-icon {
      color: var(--gray-600);
      transition: transform 0.2s ease;
      flex-shrink: 0;

      &.rotated {
        transform: rotate(-90deg);
      }
    }
  }

  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 0 10px 10px;
  }
}
</style>
