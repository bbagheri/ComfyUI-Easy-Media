import { uuid } from '@/lib/uuid'
import { mediaSlotNumber } from '@/lib/media-url'
import type { MultiTrack, MultiTrackSegment, MultiTrackSourceType, MultiTrackTaskImage } from '@/types/multitrack'

export const MAX_TASK_IMAGES = 9
export const MULTIPLE_MEDIA_SEPARATOR = '|MULTIPLE|'

export function taskImagesFromContent(images: MultiTrackTaskImage[] | undefined): MultiTrackTaskImage[] {
  return Array.isArray(images) ? images : []
}

export function activeTaskImages(images: MultiTrackTaskImage[] | undefined): MultiTrackTaskImage[] {
  return taskImagesFromContent(images).filter((image) => image.muted !== true)
}

export function taskImageReferenceIndex(
  images: MultiTrackTaskImage[],
  imageId: string,
  offset = 0,
): number | null {
  const imageIndex = images.findIndex((image) => image.id === imageId)
  if (imageIndex < 0 || images[imageIndex].muted === true) return null
  return offset + images.slice(0, imageIndex + 1).filter((image) => image.muted !== true).length - 1
}

export function taskImageSlotNumber(image: MultiTrackTaskImage, fallbackIndex: number): number {
  return mediaSlotNumber(image.slot_name, fallbackIndex)
}

export function taskImageIdentity(image: MultiTrackTaskImage): string {
  const sourceType = image.source_type ?? 'input'
  const path = sourceType === 'slot'
    ? image.slot_name
    : image.file_path ?? image.local_path ?? image.url ?? image.file_name
  return `${sourceType}:${path ?? image.id}`
}

export function canEnableSharedTaskImage(
  segments: MultiTrackSegment[],
  image: MultiTrackTaskImage,
): boolean {
  const identity = taskImageIdentity(image)
  return segments.every((segment) => {
    const images = taskImagesFromContent(segment.content.images)
    return images.some((item) => taskImageIdentity(item) === identity)
      || images.length < MAX_TASK_IMAGES
  })
}

export function sharedTaskImageUpdates(
  segments: MultiTrackSegment[],
  image: MultiTrackTaskImage,
  enabled: boolean,
): Array<{ segmentId: string; images: MultiTrackTaskImage[] }> {
  const identity = taskImageIdentity(image)
  return segments.map((segment) => {
    const images = taskImagesFromContent(segment.content.images)
    const matchingImage = images.find((item) => taskImageIdentity(item) === identity)
    if (enabled) {
      const sharedImage = matchingImage
        ? { ...matchingImage, shared_reference: true }
        : {
            ...image,
            id: uuid(),
            shared_reference: true,
            shared_reference_copy: true,
          }
      return {
        segmentId: segment.id,
        images: [
          ...images.filter((item) => item.shared_reference === true && taskImageIdentity(item) !== identity),
          sharedImage,
          ...images.filter((item) => item.shared_reference !== true && taskImageIdentity(item) !== identity),
        ].slice(0, MAX_TASK_IMAGES),
      }
    }
    return {
      segmentId: segment.id,
      images: images.flatMap((item) => {
        if (taskImageIdentity(item) !== identity) return [item]
        if (item.shared_reference_copy === true) return []
        const localImage = { ...item, shared_reference: false }
        delete localImage.shared_reference_copy
        return [localImage]
      }),
    }
  })
}

export function mergeOrUpdateTaskImages(
  existing: MultiTrackTaskImage[],
  incoming: MultiTrackTaskImage[],
  maxImages = MAX_TASK_IMAGES,
): MultiTrackTaskImage[] {
  const incomingByIdentity = new Map<string, MultiTrackTaskImage>()
  for (const item of incoming) {
    incomingByIdentity.set(taskImageIdentity(item), item)
  }

  const existingIdentities = new Set<string>()
  const updatedExisting = existing.map((item) => {
    const identity = taskImageIdentity(item)
    existingIdentities.add(identity)
    const match = incomingByIdentity.get(identity)
    if (!match) return item
    return {
      ...item,
      ...match,
      id: item.id,
      shared_reference: item.shared_reference,
      shared_reference_copy: item.shared_reference_copy,
      muted: item.muted,
      panorama_view: item.panorama_view,
      media_revision: match.media_revision ?? String(Date.now()),
    }
  })

  const appended: MultiTrackTaskImage[] = []
  const seenAdded = new Set<string>()
  for (const item of incoming) {
    const identity = taskImageIdentity(item)
    if (existingIdentities.has(identity) || seenAdded.has(identity)) continue
    seenAdded.add(identity)
    appended.push(item)
  }

  return [...updatedExisting, ...appended].slice(0, maxImages)
}

export function synchronizeSharedTaskImages(tracks: MultiTrack[]): MultiTrack[] {
  const taskSegments = tracks
    .filter((track) => track.type === 'task')
    .flatMap((track) => track.segments)
  const sharedImages = new Map<string, MultiTrackTaskImage>()
  const latestRevisions = new Map<string, string>()
  for (const segment of taskSegments) {
    for (const image of taskImagesFromContent(segment.content.images)) {
      const identity = taskImageIdentity(image)
      if (image.media_revision) {
        const prevRev = latestRevisions.get(identity)
        if (!prevRev || image.media_revision > prevRev) {
          latestRevisions.set(identity, image.media_revision)
        }
      }
      if (image.shared_reference !== true) continue
      if (!sharedImages.has(identity)) sharedImages.set(identity, image)
    }
  }
  const hasRevisionMismatch = taskSegments.some((segment) =>
    taskImagesFromContent(segment.content.images).some((image) => {
      const rev = latestRevisions.get(taskImageIdentity(image))
      return rev !== undefined && image.media_revision !== rev
    }),
  )
  if (sharedImages.size === 0 && !hasRevisionMismatch) return tracks

  return tracks.map((track) => {
    if (track.type !== 'task') return track
    return {
      ...track,
      segments: track.segments.map((segment) => {
        const images = taskImagesFromContent(segment.content.images).map((item) => {
          const rev = latestRevisions.get(taskImageIdentity(item))
          return rev !== undefined && item.media_revision !== rev
            ? { ...item, media_revision: rev }
            : item
        })
        if (sharedImages.size === 0) {
          return {
            ...segment,
            content: {
              ...segment.content,
              images: images.slice(0, MAX_TASK_IMAGES),
            },
          }
        }
        const common = Array.from(sharedImages.entries()).map(([identity, sharedImage]) => {
          const rev = latestRevisions.get(identity)
          const existing = images.find((item) => taskImageIdentity(item) === identity)
          return existing
            ? { ...existing, shared_reference: true, ...(rev ? { media_revision: rev } : {}) }
            : {
                ...sharedImage,
                ...(rev ? { media_revision: rev } : {}),
                id: uuid(),
                shared_reference: true,
                shared_reference_copy: true,
              }
        })
        const local = images.filter((item) => !sharedImages.has(taskImageIdentity(item)))
        return {
          ...segment,
          content: {
            ...segment.content,
            images: [...common, ...local].slice(0, MAX_TASK_IMAGES),
          },
        }
      }),
    }
  })
}

export function createTaskImage(
  filePath: string,
  source: MultiTrackSourceType,
  mediaRevision?: string,
): MultiTrackTaskImage {
  const normalizedSource = filePath.startsWith('__slot__:') ? 'slot' : source
  const slotName = normalizedSource === 'slot' ? filePath.replace(/^__slot__:/, '') : undefined
  const fileName = slotName ?? filePath.split(/[\\/]/).pop() ?? filePath
  return {
    id: uuid(),
    source_type: normalizedSource,
    file_path: normalizedSource === 'input' || normalizedSource === 'output' ? filePath : undefined,
    local_path: normalizedSource === 'local' ? filePath : undefined,
    url: normalizedSource === 'url' ? filePath : undefined,
    slot_name: slotName,
    file_name: fileName,
    media_revision: normalizedSource === 'slot' ? undefined : (mediaRevision ?? String(Date.now())),
  }
}

export function splitSelectedTaskMedia(value: string): string[] {
  return value.split(MULTIPLE_MEDIA_SEPARATOR).filter((item) => item.length > 0)
}

export async function uploadTaskImageFile(file: File): Promise<MultiTrackTaskImage> {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('type', 'input')
  formData.append('overwrite', 'true')
  const response = await fetch('/upload/image', {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new Error(`Failed to upload image: ${file.name}`)
  }
  const result = await response.json() as { name?: string; subfolder?: string }
  const name = result.name ?? file.name
  const subfolder = result.subfolder ?? ''
  return {
    id: uuid(),
    source_type: 'input',
    file_path: subfolder ? `${subfolder}/${name}` : name,
    file_name: name,
    media_revision: String(Date.now()),
  }
}
