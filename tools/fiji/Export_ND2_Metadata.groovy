#@ File(label = "ND2 file", style = "open", extensions = {"nd2"}) nd2File
#@ Integer(label = "Series (1-based)", min = "1", value = "1") seriesNumber
#@ String(label = "Output filename", value = "FPS_Details.xlsx") outputFilename
#@ Boolean(label = "Overwrite existing workbook", value = false) overwriteExisting

import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

import loci.formats.ImageReader
import loci.formats.MetadataTools
import ome.units.UNITS

if (!nd2File.isFile()) {
    throw new IllegalArgumentException("ND2 file does not exist: ${nd2File}")
}
if (!nd2File.name.toLowerCase(Locale.ROOT).endsWith('.nd2')) {
    throw new IllegalArgumentException("Input must be an ND2 file: ${nd2File}")
}
if (!outputFilename || new File(outputFilename).name != outputFilename ||
        !outputFilename.toLowerCase(Locale.ROOT).endsWith('.xlsx')) {
    throw new IllegalArgumentException(
        'Output filename must be a filename ending in .xlsx, not a path.'
    )
}

File outputFile = new File(nd2File.parentFile, outputFilename)
if (outputFile.exists() && !overwriteExisting) {
    throw new IllegalStateException(
        "${outputFile} already exists. Enable 'Overwrite existing workbook' " +
        'to replace it.'
    )
}

def metadata = MetadataTools.createOMEXMLMetadata()
def reader = new ImageReader()
reader.setMetadataStore(metadata)

Map<String, Object> details
try {
    // setId parses the ND2 header and metadata; it does not load the image stack.
    reader.setId(nd2File.absolutePath)

    int seriesIndex = seriesNumber - 1
    if (seriesIndex < 0 || seriesIndex >= reader.seriesCount) {
        throw new IllegalArgumentException(
            "Series must be between 1 and ${reader.seriesCount}; got ${seriesNumber}."
        )
    }
    reader.series = seriesIndex

    int frames = reader.sizeT
    if (frames < 1) {
        throw new IllegalStateException('The selected ND2 series has no time points.')
    }

    def physicalSizeX = metadata.getPixelsPhysicalSizeX(seriesIndex)
    if (physicalSizeX == null) {
        throw new IllegalStateException(
            'Bio-Formats did not find the X pixel size in the ND2 metadata.'
        )
    }
    double pixelScaleMicrometres =
        physicalSizeX.value(UNITS.MICROMETRE).doubleValue()
    requirePositiveFinite(pixelScaleMicrometres, 'Pixel Scale Factor')

    Double firstTimestamp = findTimestampSeconds(
        metadata, reader, seriesIndex, 0
    )
    Double lastTimestamp = findTimestampSeconds(
        metadata, reader, seriesIndex, frames - 1
    )
    Double timeIncrement = quantityInSeconds(
        metadata.getPixelsTimeIncrement(seriesIndex)
    )

    if (firstTimestamp == null && lastTimestamp == null && timeIncrement != null) {
        firstTimestamp = 0.0d
        lastTimestamp = (frames - 1) * timeIncrement
    } else if (firstTimestamp == null && lastTimestamp != null &&
            timeIncrement != null) {
        firstTimestamp = lastTimestamp - ((frames - 1) * timeIncrement)
    } else if (lastTimestamp == null && firstTimestamp != null &&
            timeIncrement != null) {
        lastTimestamp = firstTimestamp + ((frames - 1) * timeIncrement)
    }

    if (firstTimestamp == null || lastTimestamp == null) {
        throw new IllegalStateException(
            'Bio-Formats did not find first/last plane timestamps or a time ' +
            'increment in the ND2 metadata.'
        )
    }
    requireFinite(firstTimestamp, 'Timestamp #First')
    requireFinite(lastTimestamp, 'Timestamp #Last')

    double elapsedSeconds = lastTimestamp - firstTimestamp
    double fps
    if (elapsedSeconds > 0.0d) {
        // This intentionally matches the existing hand-authored workbooks.
        fps = frames / elapsedSeconds
    } else if (frames == 1 && timeIncrement != null && timeIncrement > 0.0d) {
        elapsedSeconds = timeIncrement
        lastTimestamp = firstTimestamp + timeIncrement
        fps = 1.0d / timeIncrement
    } else {
        throw new IllegalStateException(
            'The last timestamp must be later than the first timestamp.'
        )
    }
    requirePositiveFinite(fps, 'FPS')

    details = [
        resolution: "${reader.sizeX}x${reader.sizeY}",
        frames: frames,
        pixelScaleFactor: pixelScaleMicrometres,
        scaleUnits: 'um',
        timestampFirst: firstTimestamp,
        timestampLast: lastTimestamp,
        elapsedSeconds: elapsedSeconds,
        fps: fps,
        seriesCount: reader.seriesCount,
        selectedSeries: seriesNumber
    ]
} finally {
    reader.close()
}

writeWorkbookAtomically(outputFile, details)

println "Saved ND2 capture metadata to: ${outputFile.absolutePath}"
println "Series ${details.selectedSeries}/${details.seriesCount}: " +
    "${details.frames} frames at ${details.resolution}, ${details.fps} FPS, " +
    "pixel scale ${details.pixelScaleFactor} ${details.scaleUnits}."

Double findTimestampSeconds(metadata, reader, int seriesIndex, int timepoint) {
    int preferredPlane = reader.getIndex(0, 0, timepoint)
    Double preferredTimestamp = planeTimestampSeconds(
        metadata, seriesIndex, preferredPlane, timepoint
    )
    if (preferredTimestamp != null) {
        return preferredTimestamp
    }

    int planeCount = metadata.getPlaneCount(seriesIndex)
    for (int plane = 0; plane < planeCount; plane++) {
        Double timestamp = planeTimestampSeconds(
            metadata, seriesIndex, plane, timepoint
        )
        if (timestamp != null) {
            return timestamp
        }
    }
    return null
}

Double planeTimestampSeconds(metadata, int seriesIndex, int plane,
        int expectedTimepoint) {
    if (plane < 0 || plane >= metadata.getPlaneCount(seriesIndex)) {
        return null
    }
    def planeT = metadata.getPlaneTheT(seriesIndex, plane)
    if (planeT != null && planeT.value != expectedTimepoint) {
        return null
    }
    return quantityInSeconds(metadata.getPlaneDeltaT(seriesIndex, plane))
}

Double quantityInSeconds(quantity) {
    return quantity == null ? null : quantity.value(UNITS.SECOND).doubleValue()
}

void requireFinite(double value, String label) {
    if (Double.isNaN(value) || Double.isInfinite(value)) {
        throw new IllegalStateException("${label} is not finite: ${value}")
    }
}

void requirePositiveFinite(double value, String label) {
    requireFinite(value, label)
    if (value <= 0.0d) {
        throw new IllegalStateException("${label} must be positive: ${value}")
    }
}

void writeWorkbookAtomically(File outputFile, Map<String, Object> details) {
    File temporaryFile = new File(
        outputFile.parentFile,
        ".${outputFile.name}.${UUID.randomUUID()}.tmp"
    )
    try {
        writeWorkbook(temporaryFile, details)
        try {
            Files.move(
                temporaryFile.toPath(),
                outputFile.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING
            )
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(
                temporaryFile.toPath(),
                outputFile.toPath(),
                StandardCopyOption.REPLACE_EXISTING
            )
        }
    } finally {
        Files.deleteIfExists(temporaryFile.toPath())
    }
}

void writeWorkbook(File file, Map<String, Object> details) {
    def entries = [
        '[Content_Types].xml': contentTypesXml(),
        '_rels/.rels': packageRelationshipsXml(),
        'xl/workbook.xml': workbookXml(),
        'xl/_rels/workbook.xml.rels': workbookRelationshipsXml(),
        'xl/styles.xml': stylesXml(),
        'xl/worksheets/sheet1.xml': worksheetXml(details)
    ]

    new ZipOutputStream(new FileOutputStream(file)).withCloseable { zip ->
        entries.each { String path, String xml ->
            zip.putNextEntry(new ZipEntry(path))
            zip.write(xml.getBytes(StandardCharsets.UTF_8))
            zip.closeEntry()
        }
    }
}

String worksheetXml(Map<String, Object> details) {
    String resolution = xmlEscape(details.resolution.toString())
    String units = xmlEscape(details.scaleUnits.toString())
    return xmlHeader() +
        '<worksheet xmlns="http://schemas.openxmlformats.org/' +
        'spreadsheetml/2006/main">' +
        '<dimension ref="A1:B9"/><sheetViews><sheetView workbookViewId="0"/>' +
        '</sheetViews><sheetFormatPr defaultRowHeight="15"/>' +
        '<cols><col min="1" max="1" width="20" customWidth="1"/>' +
        '<col min="2" max="2" width="18" customWidth="1"/></cols>' +
        '<sheetData>' +
        row(1, inlineCell('A1', 'ND2 Frames', 1)) +
        row(2, inlineCell('A2', 'Resolution') + inlineCell('B2', resolution)) +
        row(3, inlineCell('A3', 'Frames') + numberCell('B3', details.frames)) +
        row(4, inlineCell('A4', 'Pixel Scale Factor') +
            numberCell('B4', details.pixelScaleFactor)) +
        row(5, inlineCell('A5', 'Scale Units') + inlineCell('B5', units)) +
        row(6, inlineCell('A6', 'Timestamp #First:') +
            numberCell('B6', details.timestampFirst)) +
        row(7, inlineCell('A7', 'Timestamp #Last:') +
            numberCell('B7', details.timestampLast)) +
        row(8, inlineCell('A8', 'Δt (s):') +
            formulaCell('B8', 'B7-B6', details.elapsedSeconds)) +
        row(9, inlineCell('A9', 'FPS:') +
            formulaCell('B9', 'B3/B8', details.fps)) +
        '</sheetData><mergeCells count="1"><mergeCell ref="A1:B1"/>' +
        '</mergeCells><pageMargins left="0.7" right="0.7" top="0.75" ' +
        'bottom="0.75" header="0.3" footer="0.3"/></worksheet>'
}

String row(int index, String cells) {
    return "<row r=\"${index}\">${cells}</row>"
}

String inlineCell(String reference, String value, int style = 0) {
    return "<c r=\"${reference}\" s=\"${style}\" t=\"inlineStr\">" +
        "<is><t>${value}</t></is></c>"
}

String numberCell(String reference, Object value) {
    return "<c r=\"${reference}\"><v>${value}</v></c>"
}

String formulaCell(String reference, String formula, Object cachedValue) {
    return "<c r=\"${reference}\"><f>${formula}</f>" +
        "<v>${cachedValue}</v></c>"
}

String xmlEscape(String value) {
    return value.replace('&', '&amp;').replace('<', '&lt;')
        .replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
}

String xmlHeader() {
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
}

String contentTypesXml() {
    return xmlHeader() +
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-' +
        'package.relationships+xml"/><Default Extension="xml" ' +
        'ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ' +
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.' +
        'sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ' +
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.' +
        'worksheet+xml"/><Override PartName="/xl/styles.xml" ' +
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.' +
        'styles+xml"/></Types>'
}

String packageRelationshipsXml() {
    return xmlHeader() +
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/' +
        'relationships"><Relationship Id="rId1" Type="http://schemas.' +
        'openxmlformats.org/officeDocument/2006/relationships/officeDocument" ' +
        'Target="xl/workbook.xml"/></Relationships>'
}

String workbookXml() {
    return xmlHeader() +
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" ' +
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/' +
        'relationships"><sheets><sheet name="FPS Details" sheetId="1" ' +
        'r:id="rId1"/></sheets><calcPr calcMode="auto"/></workbook>'
}

String workbookRelationshipsXml() {
    return xmlHeader() +
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/' +
        'relationships"><Relationship Id="rId1" Type="http://schemas.' +
        'openxmlformats.org/officeDocument/2006/relationships/worksheet" ' +
        'Target="worksheets/sheet1.xml"/><Relationship Id="rId2" ' +
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/' +
        'relationships/styles" Target="styles.xml"/></Relationships>'
}

String stylesXml() {
    return xmlHeader() +
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/' +
        'main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/>' +
        '</font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>' +
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>' +
        '<borders count="1"><border/></borders><cellStyleXfs count="1">' +
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>' +
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" ' +
        'borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" ' +
        'borderId="0" xfId="0" applyFont="1"/></cellXfs>' +
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" ' +
        'builtinId="0"/></cellStyles></styleSheet>'
}
